# main.py
# 目的：CoinDesk/CoinPost → Gemini要約 → SQLite保存 → HTML/JSON/RSSで配信（テンプレート分離版）
# 依存：fastapi, uvicorn, SQLAlchemy, feedparser, httpx, selectolax, APScheduler, google-generativeai, Jinja2

import os
import re
import html as htmlesc
import asyncio
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel

from sqlalchemy import (
    create_engine, String, Integer, Text, DateTime,
    UniqueConstraint, select, desc, func
)
from sqlalchemy.orm import declarative_base, Mapped, mapped_column, sessionmaker

import feedparser
import httpx
from selectolax.parser import HTMLParser
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# =========================
# 環境変数
# =========================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
TARGET_LANG = os.getenv("TARGET_LANG", "ja")
SUMMARY_TONE = os.getenv("SUMMARY_TONE", "投資家向けに、価格影響と重要ファクトを3〜5行で要約")
SITE_TITLE = os.getenv("SITE_TITLE", "Crypto News Agent")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

if not GOOGLE_API_KEY:
    print("⚠️ GOOGLE_API_KEY が未設定です。要約はフォールバックになります。")

# =========================
# Gemini クライアント
# =========================
try:
    import google.generativeai as genai
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        gemini_model = genai.GenerativeModel(
            model_name=GEMINI_MODEL_NAME,
            generation_config=generation_config
        )
    else:
        gemini_model = None
except Exception:
    gemini_model = None

SYSTEM_PROMPT = f"""あなたは暗号資産ニュースの専門アナリストです。
出力は{TARGET_LANG}。煽りは禁止。数値・日付・固有名詞は正確に。
企業・トークン・規制・相場影響（短期/中期/長期の目線）を簡潔に整理してください。"""

def summarize_by_gemini(title: str, url: str, raw_snippet: str) -> str:
    """Geminiで日本語要約（失敗時は簡易フォールバック）"""
    user_prompt = f"""次の記事を{TARGET_LANG}で{SUMMARY_TONE}。
- タイトル: {title}
- URL: {url}
- 抜粋/説明: {raw_snippet or '（抜粋なし）'}
- 形式: 箇条書き3-5点 + 最後に「一言見立て：...」
"""
    try:
        if not gemini_model:
            raise RuntimeError("Gemini client not ready")
        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
        resp = gemini_model.generate_content(full_prompt)
        text = (resp.text or "").strip()
        if not text:
            raise ValueError("Empty response")
        return text
    except Exception:
        return f"- {title}\n- {url}\n一言見立て：詳細不明。リンク先を参照。"

# =========================
# DB（SQLite / SQLAlchemy 2.x）
# =========================
Base = declarative_base()
engine = create_engine("sqlite:///news.db", future=True, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

class Article(Base):
    __tablename__ = "articles"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(50), index=True)  # "coindesk" / "coinpost"
    title: Mapped[str] = mapped_column(String(512))
    url: Mapped[str] = mapped_column(String(1024), unique=True)
    published_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    summary_ja: Mapped[str] = mapped_column(Text)
    categories: Mapped[str] = mapped_column(String(256), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint('url', name='uq_article_url'),)

def init_db():
    Base.metadata.create_all(bind=engine)

# =========================
# ユーティリティ
# =========================
def parse_pubdate(dt) -> datetime:
    try:
        from email.utils import parsedate_to_datetime
        d = parsedate_to_datetime(str(dt))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CryptoNewsAgent/1.0)"}

# =========================
# フィード取得（RSS優先、CoinPostはフォールバック）
# =========================
COINDESK_RSS = "https://www.coindesk.com/arc/outboundfeeds/rss/"
COINPOST_RSS_CANDIDATE = "https://coinpost.jp/?feed=rss2"
COINPOST_HOME = "https://coinpost.jp/"

async def fetch_rss_entries(url: str):
    async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:
        r = await client.get(url)
        r.raise_for_status()
    d = feedparser.parse(r.text)
    for e in d.entries:
        yield {
            "title": (e.get("title") or "").strip(),
            "link": (e.get("link") or "").strip(),
            "published": parse_pubdate(e.get("published") or e.get("updated") or datetime.now(timezone.utc)),
            "summary": (e.get("summary") or "").strip(),
            "tags": ",".join(t["term"] for t in e.get("tags", []) if t.get("term"))
        }

async def fetch_coindesk():
    async for item in fetch_rss_entries(COINDESK_RSS):
        item["source"] = "coindesk"
        yield item

async def fetch_coinpost():
    # 1) RSSを試す
    try:
        async for item in fetch_rss_entries(COINPOST_RSS_CANDIDATE):
            item["source"] = "coinpost"
            yield item
        return
    except Exception:
        pass
    # 2) HTMLフォールバック（トップ新着を軽量抽出）
    async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:
        r = await client.get(COINPOST_HOME)
        r.raise_for_status()
    htmlp = HTMLParser(r.text)
    seen = set()
    for a in htmlp.css("a"):
        href = a.attributes.get("href", "")
        title = a.text(strip=True)
        if not href or not title:
            continue
        if not href.startswith("https://coinpost.jp/"):
            continue
        # 記事URLらしいパターン（/p/123456/ か /YYYY/MM/DD/.../ など）
        if not re.search(r"/\d+/$|/p/\d+", href):
            continue
        if href in seen:
            continue
        seen.add(href)
        yield {
            "source": "coinpost",
            "title": title[:200],
            "link": href,
            "published": datetime.now(timezone.utc),  # 厳密な時刻は詳細取得で上書き可
            "summary": "",
            "tags": ""
        }

# =========================
# ハーベスト & 要約（保存）
# =========================
async def harvest_and_summarize():
    sources = [fetch_coindesk(), fetch_coinpost()]
    for agen in sources:
        async for item in agen:
            url = item["link"]
            if not url:
                continue
            with SessionLocal() as s:
                exists = s.scalar(select(Article).where(Article.url == url).limit(1))
                if exists:
                    continue
                summary = summarize_by_gemini(item["title"], url, item.get("summary", ""))
                a = Article(
                    source=item["source"],
                    title=item["title"],
                    url=url,
                    published_at=item["published"],
                    summary_ja=summary,
                    categories=item.get("tags", "")
                )
                s.add(a)
                try:
                    s.commit()
                except Exception:
                    s.rollback()

# =========================
# FastAPI（API + HTML + RSS）
# =========================
class ArticleOut(BaseModel):
    id: int
    source: str
    title: str
    url: str
    published_at: datetime
    summary_ja: str
    categories: str
    class Config:
        from_attributes = True

app = FastAPI(title=SITE_TITLE)

# テンプレート & 静的ファイル
templates = Jinja2Templates(directory="templates")
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def on_startup():
    init_db()
    # 初回収集（非同期で開始）
    asyncio.create_task(harvest_and_summarize())
    # 5分おきに実行（Renderでスケール1台推奨／複数台ならCronへ移行）
    scheduler = AsyncIOScheduler()
    scheduler.add_job(harvest_and_summarize, "interval", minutes=5, next_run_time=None)
    scheduler.start()
    app.state.scheduler = scheduler

@app.get("/health")
async def health():
    return {"ok": True, "time": datetime.utcnow().isoformat() + "Z"}

@app.get("/news", response_model=List[ArticleOut])
def list_news(
    source: Optional[str] = Query(None),
    limit: int = Query(30, ge=1, le=200),
    page: int = Query(1, ge=1)
):
    offset = (page - 1) * limit
    with SessionLocal() as s:
        base = select(Article)
        if source:
            base = base.where(Article.source == source)
        stmt = base.order_by(desc(Article.published_at)).offset(offset).limit(limit)
        items = s.scalars(stmt).all()
        return items

@app.get("/", response_class=HTMLResponse)
def homepage(
    request: Request,
    source: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=5, le=100),
    auto: int = Query(60, ge=0, le=600)
):
    offset = (page - 1) * limit
    with SessionLocal() as s:
        base = select(Article)
        if source:
            base = base.where(Article.source == source)
        total = s.scalar(select(func.count()).select_from(base.subquery()))
        stmt = base.order_by(desc(Article.published_at)).offset(offset).limit(limit)
        rows = s.scalars(stmt).all()

    total_pages = max((total + limit - 1) // limit, 1)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "SITE_TITLE": SITE_TITLE,
            "rows": rows,
            "source": source,
            "page": page,
            "limit": limit,
            "auto": auto,
            "total_pages": total_pages,
        }
    )

@app.get("/rss.xml")
def rss(
    feed_title: str = SITE_TITLE,
    feed_link: str = "",
    feed_desc: str = "最新の要約済みクリプトニュース"
):
    with SessionLocal() as s:
        stmt = select(Article).order_by(desc(Article.published_at)).limit(50)
        items = s.scalars(stmt).all()

    now = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
    link = feed_link or "http://localhost:8000/"
    def esc(x): return htmlesc.escape(str(x)) if x is not None else ""

    items_xml = []
    for a in items:
        pub = a.published_at.strftime("%a, %d %b %Y %H:%M:%S GMT")
        items_xml.append(f"""
    <item>
      <title>{esc(a.title)}</title>
      <link>{esc(a.url)}</link>
      <guid isPermaLink="false">{esc(a.url)}</guid>
      <pubDate>{pub}</pubDate>
      <description>{esc(a.summary_ja)}</description>
      <category>{esc(a.source)}</category>
    </item>""")

    rss_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
 <channel>
  <title>{esc(feed_title)}</title>
  <link>{esc(link)}</link>
  <description>{esc(feed_desc)}</description>
  <language>ja</language>
  <lastBuildDate>{now}</lastBuildDate>
  {''.join(items_xml)}
 </channel>
</rss>"""
    return Response(content=rss_xml, media_type="application/rss+xml")

# ローカル起動用
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
