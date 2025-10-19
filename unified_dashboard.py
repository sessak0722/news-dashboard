# unified_dashboard.py — 집계→CSV→대시보드 (NAVER × GDELT)
# + 안전가드/‘처음 등장’ 버그픽스/신선도배지/헬스체크 포함
# 실행: streamlit run unified_dashboard.py

import os, time, math, json, hashlib, requests, random, re
from datetime import datetime, timedelta, timezone, date
from dateutil import parser as dtparser
from dotenv import load_dotenv
from urllib.parse import quote, urlparse
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass

# ── 추가: 실시간 주가 ─────────────────────────────────────────────────────────
try:
    import yfinance as yf
except Exception:
    yf = None  # 설치 전에도 대시보드가 죽지 않도록

# ───────────────────────── 기본 설정 ─────────────────────────
st.set_page_config(page_title="뉴스 언급 급등 통합 대시보드", layout="wide")
load_dotenv()
CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
KST = timezone(timedelta(hours=9))

ENTITIES_CSV = "entities.csv"
OUT_CSV = "cross_batch_counts.csv"
DOMAIN_WEIGHTS_JSON = "domain_weights.json"
KR_HOLIDAY_CSV = os.getenv("KR_HOLIDAY_CSV", "").strip()  # (선택) YYYY-MM-DD 한 줄 하나

# 429 회피 파라미터
NAVER_CALL_DELAY_SEC = float(os.getenv("NAVER_CALL_DELAY_SEC", "0.2"))
NAVER_MAX_RETRY = int(os.getenv("NAVER_MAX_RETRY", "5"))
NAVER_BACKOFF_BASE = float(os.getenv("NAVER_BACKOFF_BASE", "0.8"))
NAVER_BACKOFF_CAP = float(os.getenv("NAVER_BACKOFF_CAP", "8.0"))

# 캐시 디렉터리
CACHE_DIR = Path(".cache"); CACHE_NAVER = CACHE_DIR / "naver"; CACHE_GDELT = CACHE_DIR / "gdelt"
for p in (CACHE_NAVER, CACHE_GDELT): p.mkdir(parents=True, exist_ok=True)

# requests 세션
SESSION = requests.Session()
ADAPTER = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=2)
SESSION.mount("http://", ADAPTER); SESSION.mount("https://", ADAPTER)
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

# ───────────────────────── 공용 유틸 ─────────────────────────
def today_tag(): return datetime.now(KST).strftime("%Y%m%d")

def cache_key(*parts) -> str:
    raw = "||".join(str(p) for p in parts); return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def cache_get(path: Path, max_age_hours: int):
    if not path.exists(): return None
    try:
        age = datetime.now(KST) - datetime.fromtimestamp(path.stat().st_mtime, KST)
        if age > timedelta(hours=max_age_hours): return None
        with path.open("r", encoding="utf-8") as f: return json.load(f)
    except Exception: return None

def cache_put(path: Path, obj):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False)
    except Exception: pass

def strip_html(text: str) -> str:
    if not text: return ""
    text = re.sub(r"</?b>", "", text); text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[^;\s]+;", " ", text); text = re.sub(r"\s+", " ", text).strip()
    return text

def short_summary(text: str, limit: int = 140) -> str:
    s = strip_html(text); return (s[:limit] + "…") if len(s) > limit else s

# ★★★ TZ 안전 유틸(NEW): tz-aware/naive 혼재 인덱스 통일 패치
def to_naive_kst(idx) -> pd.DatetimeIndex:
    """
    tz-aware → Asia/Seoul로 변환 후 tz정보 제거
    tz-naive → 그대로 DatetimeIndex로 정규화
    """
    _idx = pd.DatetimeIndex(idx)
    if getattr(_idx, "tz", None) is not None:
        try:
            _idx = _idx.tz_convert(KST).tz_localize(None)
        except Exception:
            _idx = _idx.tz_localize(None)
    return _idx

# ───────────────────────── 거래일 캘린더 ─────────────────────────
@st.cache_data(show_spinner=False, ttl=12*60*60)
def load_kr_holidays(csv_path: str) -> set[date]:
    if not csv_path or not os.path.exists(csv_path): return set()
    try:
        s = pd.read_csv(csv_path, header=None, names=["d"], dtype=str)["d"].dropna().str.strip()
        return set(pd.to_datetime(s).dt.date.tolist())
    except Exception:
        return set()

def business_days_index(start: date, end: date, holidays: set[date]) -> pd.DatetimeIndex:
    # tz-aware로 만들었다가 tz 제거 → naive KST 축
    all_days = pd.date_range(start=start, end=end, freq="D", tz=KST).tz_convert(None)
    bdays = all_days[all_days.weekday < 5]
    if holidays:
        mask = ~bdays.date.astype("O").isin(holidays)
        bdays = bdays[mask]
    return pd.DatetimeIndex(bdays)

# ───────────────────────── 엔티티 로딩 ─────────────────────────
def load_entities(path: str):
    if not os.path.exists(path):
        st.error(f"{path} 파일이 없습니다. 먼저 entities.csv를 만들어주세요."); st.stop()
    _df = None
    for enc in ("utf-8-sig", "cp949", "utf-8"):
        try: _df = pd.read_csv(path, encoding=enc); break
        except Exception: continue
    if _df is None:
        st.error("entities.csv 인코딩 오류: UTF-8(또는 CSV UTF-8)로 저장 후 재시도"); st.stop()
    _df.columns = [str(c).strip().lower() for c in _df.columns]
    def colget(cols):
        for c in cols:
            if c in _df.columns: return c
        return None
    aliases = {
        "name_kr": ["name_kr","name","회사명","기업","종목명"],
        "name_en": ["name_en","english","영문명"],
        "display": ["display","표시명","보여줄이름","label","title"],
        "type":    ["type","분류","카테고리"],
        "enabled": ["enabled","use","사용","active"],
        "ticker":  ["ticker","티커","symbol","종목코드","code"],
        "aliases": ["aliases","별칭","동의어","aka","synonyms"],
    }
    for k, al in aliases.items():
        if k not in _df.columns:
            c = colget(al); _df[k] = _df[c] if c else ""
    def truthy(x): return 1 if str(x).strip().lower() in ("1","true","y","yes","on") else 0
    _df["enabled"] = _df["enabled"].map(truthy).fillna(1).astype(int)
    for c in ["name_kr","name_en","display","ticker","type","aliases"]:
        _df[c] = _df[c].fillna("").astype(str).str.strip()
    _df.loc[_df["type"]=="","type"] = "company"
    _df.loc[_df["display"]=="","display"] = _df["name_kr"].where(_df["name_kr"]!="", _df["name_en"])
    _df = _df[_df["enabled"]==1].copy()
    rows = []
    for _, r in _df.iterrows():
        if not (str(r.get("name_kr","")).strip() or str(r.get("name_en","")).strip() or str(r.get("display","")).strip()):
            continue
        rows.append({
            "type": r.get("type","company"),
            "name_kr": r.get("name_kr",""),
            "name_en": r.get("name_en",""),
            "display": r.get("display",""),
            "ticker": r.get("ticker",""),
            "aliases": r.get("aliases",""),
        })
    if not rows:
        st.error("entities.csv에 enabled=1인 엔티티가 없습니다."); st.stop()
    return rows

# ───────────────────────── 도메인 가중치 ─────────────────────────
@st.cache_data(show_spinner=False, ttl=12*60*60)
def load_domain_weights(path: str) -> dict:
    try:
        if not os.path.exists(path): return {}
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return {str(k).strip().lower(): float(v) for k, v in obj.items()}
    except Exception:
        return {}

def get_domain_weight(domain: str, wmap: dict) -> float:
    if not domain: return 1.0
    d = domain.lower()
    return float(wmap.get(d, 1.0))

# ───────────────────────── NAVER ─────────────────────────
def _sleep_with_jitter(base: float):
    time.sleep(base + random.random() * 0.15)

def naver_news_search_all(query: str, max_items: int = 100, sort: str = "date", cache_hours: int = 6):
    if not CLIENT_ID or not CLIENT_SECRET:
        raise RuntimeError("NAVER 키가 필요합니다(.env).")
    display = min(100, max(10, max_items))
    params = {"query": query, "display": display, "start": 1, "sort": sort}
    key = cache_key(today_tag(), query, display, sort)
    cpath = CACHE_NAVER / f"{key}.json"
    cached = cache_get(cpath, cache_hours)
    if cached is not None: return cached
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {"X-Naver-Client-Id": CLIENT_ID, "X-Naver-Client-Secret": CLIENT_SECRET}
    last_err = None
    for attempt in range(1, NAVER_MAX_RETRY + 1):
        try:
            r = SESSION.get(url, headers=headers, params=params, timeout=15)
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                wait = float(ra) if ra and re.match(r"^\d+(\.\d+)?$", ra) else min(NAVER_BACKOFF_CAP, NAVER_BACKOFF_BASE * (2 ** (attempt - 1)))
                _sleep_with_jitter(wait); last_err = requests.HTTPError(f"429 Too Many Requests (attempt {attempt})")
                continue
            if 500 <= r.status_code < 600:
                wait = min(NAVER_BACKOFF_CAP, NAVER_BACKOFF_BASE * (2 ** (attempt - 1)))
                _sleep_with_jitter(wait); last_err = requests.HTTPError(f"{r.status_code} Server Error (attempt {attempt})")
                continue
            r.raise_for_status()
            items = r.json().get("items", [])
            cache_put(cpath, items); _sleep_with_jitter(NAVER_CALL_DELAY_SEC)
            return items
        except requests.RequestException as e:
            last_err = e
            wait = min(NAVER_BACKOFF_CAP, NAVER_BACKOFF_BASE * (2 ** (attempt - 1))); _sleep_with_jitter(wait)
    st.warning(f"네이버 뉴스 API 호출이 일시 제한될 수 있어요. (query='{query}')")
    return []

def extract_domain(u: str) -> str:
    if not u: return ""
    try:
        netloc = urlparse(u).netloc.lower()
        if netloc.startswith("www."): netloc = netloc[4:]
        return netloc
    except Exception: return ""

def count_in_windows_and_domains(items, now_kst: datetime, wmap: dict):
    c24 = c7 = c30 = 0
    c24_w = c7_w = 0.0
    domains_24h = set()
    for it in items:
        pub = it.get("pubDate"); link = it.get("originallink") or it.get("link")
        try:
            dt = dtparser.parse(pub)
            if not dt.tzinfo: dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(KST)
        except Exception:
            continue
        d = extract_domain(link); w = get_domain_weight(d, wmap)
        delta = now_kst - dt
        if delta <= timedelta(days=1):
            c24 += 1; c24_w += w
            if d: domains_24h.add(d)
        if delta <= timedelta(days=7):
            c7 += 1; c7_w += w
        if delta <= timedelta(days=30):
            c30 += 1
    return c24, c7, c30, len(domains_24h), round(c24_w,3), round(c7_w,3)

def spike_from_7d(c24: float, c7_total: float):
    baseline_per24 = max(1e-4, c7_total / 7.0)
    return c24 / baseline_per24

# ───────────────────────── GDELT ─────────────────────────
def _gdelt_timeline_total(query: str, timespan: str, cache_hours: int = 6) -> int | None:
    key = cache_key(today_tag(), query, timespan); cpath = CACHE_GDELT / f"{key}.json"
    cached = cache_get(cpath, cache_hours)
    if cached is not None: return int(cached)
    url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={quote(query)}&mode=TimelineVol&format=json&timespan={timespan}"
    for attempt in range(1, 3 + 1):
        try:
            r = SESSION.get(url, timeout=25)
            if r.status_code != 200: time.sleep(0.8*attempt); continue
            try: js = r.json()
            except Exception: time.sleep(0.8*attempt); continue
            timeline = js.get("timeline", [])
            if not timeline: cache_put(cpath, 0); return 0
            total = 0
            for b in timeline[0].get("data", []):
                try: total += int(b.get("value", 0))
                except Exception: pass
            cache_put(cpath, total); return total
        except requests.RequestException:
            time.sleep(0.8*attempt)
    return None

def _gdelt_timeline_series(query: str, timespan: str = "60d", cache_hours: int = 6) -> pd.Series:
    key = cache_key("series", today_tag(), query, timespan); cpath = CACHE_GDELT / f"{key}.json"
    cached = cache_get(cpath, cache_hours)
    if cached is not None:
        try:
            s = pd.Series(cached).astype(int); s.index = pd.to_datetime(s.index); return s
        except Exception: pass
    url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={quote(query)}&mode=TimelineVol&format=json&timespan={timespan}"
    try:
        r = SESSION.get(url, timeout=25)
        if r.status_code != 200: return pd.Series(dtype=int)
        js = r.json(); timeline = js.get("timeline", [])
        if not timeline: cache_put(cpath, {}); return pd.Series(dtype=int)
        data = timeline[0].get("data", [])
        pairs = {}
        for b in data:
            try: pairs[b["date"]] = int(b.get("value", 0))
            except Exception: continue
        cache_put(cpath, pairs)
        s = pd.Series(pairs).astype(int); s.index = pd.to_datetime(s.index); return s
    except Exception:
        return pd.Series(dtype=int)

def gdelt_doc_count_smart(name_kr: str | None, name_en: str | None, timespan: str, cache_hours: int = 6) -> int:
    kr = (name_kr or "").strip(); en = (name_en or "").strip()
    candidates = []
    if en: candidates += [f"\"{en}\"", en]
    if kr: candidates += [f"\"{kr}\"", kr]
    for q in candidates:
        res = _gdelt_timeline_total(q, timespan, cache_hours=cache_hours)
        if res is not None: return int(res)
    return 0

# ───────────────────────── 테마 규칙 ─────────────────────────
THEME_RULES = {
    "2차전지": ["2차전지","배터리","양극재","음극재","전해질","LFP","NCM","리튬","코발트","니켈","separator","cathode","anode"],
    "원전/SMR": ["원전","SMR","소형모듈원자로","가압수","원자력","사용후핵연료","두산에너빌리티","NuScale","Westinghouse"],
    "반도체": ["반도체","파운드리","메모리","HBM","EUV","ASML","TSMC","fab","칩"],
    "AI/데이터센터": ["AI","인공지능","데이터센터","GPU","엔비디아","NVIDIA","LLM","서버","클라우드"],
    "로봇": ["로봇","AMR","AGV","로보틱스","협동로봇","actuator"],
    "전기차": ["전기차","EV","충전소","초급속","테슬라","BYD"],
    "방산": ["방산","군수","미사일","K2","K9","FA-50","defense","탄약"],
    "바이오": ["바이오","임상","FDA","신약","세포치료","mRNA","진단"],
    "해운/조선": ["해운","운임","컨테이너","조선","LNG선","VLCC","선박"],
    "건설/인프라": ["건설","인프라","SOC","토목","원가","수주","PF"],
    "항공우주": ["항공","우주","위성","발사체","로켓","저궤도","SpaceX"],
    "통신/네트워크": ["5G","6G","광케이블","네트워크","통신사","스몰셀"],
    "에너지/정유": ["정유","석유","가스","LNG","재생에너지","풍력","태양광"],
    "희토류/자원": ["희토류","니오디뮴","텅스텐","구리","철광석","리튬광산"],
    "HVDC/전력": ["HVDC","초고압","변환소","케이블","전력망","계통"],
    "디스플레이/광학": ["OLED","LCD","디스플레이","광학","마이크로OLED","광학렌즈"],
}

def classify_themes(texts: list[str], top_k: int = 3):
    score = {k:0 for k in THEME_RULES.keys()}
    joined = " ".join([t for t in texts if t])
    for theme, kws in THEME_RULES.items():
        s = 0
        for kw in kws:
            s += 2 * len(re.findall(rf"\b{re.escape(kw)}\b", joined, flags=re.IGNORECASE))
            s += 1 * joined.lower().count(kw.lower())
        score[theme] = s
    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
    ranked = [t for t in ranked if t[1] > 0]
    return [t[0] for t in ranked[:top_k]] if ranked else []

# ───────────────────────── 기사 카드 ─────────────────────────
@st.cache_data(show_spinner=False, ttl=2*60*60)
def extract_news_cards(query_display: str, limit: int = 10, cache_hours: int = 2):
    items = naver_news_search_all(query_display, max_items=limit, sort="date", cache_hours=cache_hours)
    cards = []
    for it in items[:limit]:
        title = strip_html(it.get("title","")); desc  = short_summary(it.get("description",""))
        link  = it.get("originallink") or it.get("link"); media = extract_domain(link)
        pub   = it.get("pubDate")
        try:
            dt = dtparser.parse(pub)
            if not dt.tzinfo: dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(KST); pub_str = dt.strftime("%m-%d %H:%M")
        except Exception: pub_str = ""
        cards.append({"time": pub_str, "media": media, "title": title, "summary": desc, "link": link})
    return cards

# ───────────────────────── 집계(가중/필터 반영) ─────────────────────────
def generate_cross_csv(
    entities, out_path: str,
    fast_mode: bool, gdelt_top_k: int, skip_gdelt: bool,
    min_unique_domains: int, naver_cache_h: int, gdelt_cache_h: int
) -> pd.DataFrame:
    now = datetime.now(KST); NEW_N = 3; rows = []
    wmap = load_domain_weights(DOMAIN_WEIGHTS_JSON)

    p1 = st.progress(0, text="1/2 NAVER 집계 중...")
    total = len(entities)
    for idx, e in enumerate(entities, start=1):
        name_kr, name_en, disp = e["name_kr"], e["name_en"], e["display"]
        items = naver_news_search_all(disp, max_items=100, sort="date", cache_hours=naver_cache_h)
        n24, n7, n30, uniq24, n24_w, n7_w = count_in_windows_and_domains(items, now, wmap)
        spike_n = spike_from_7d(n24_w if n24_w>0 else n24, n7_w if n7_w>0 else n7)

        rows.append({
            "display": disp, "type": e["type"], "name_kr": name_kr, "name_en": name_en, "ticker": e.get("ticker",""),
            "naver_24h": n24, "naver_7d": n7, "naver_30d": n30,
            "naver_24h_weighted": n24_w, "naver_7d_weighted": n7_w,
            "naver_spike": round(spike_n, 3), "unique_domains_24h": uniq24,
            "gdelt_24h": 0, "gdelt_7d": 0, "gdelt_30d": 0, "gdelt_spike": 0.0,
            "combined_spike": round(spike_n, 3),
            "newly_appeared": (n30 <= 1 and n24 >= NEW_N)
        })
        if idx % 10 == 0: _sleep_with_jitter(NAVER_CALL_DELAY_SEC)
        p1.progress(min(int(idx/total*100), 100), text=f"1/2 NAVER 집계 중... ({idx}/{total})")
    p1.empty()

    df = pd.DataFrame(rows)
    if min_unique_domains > 0:
        mask = df["unique_domains_24h"] >= min_unique_domains
        df.loc[~mask, ["combined_spike","naver_spike"]] = 0

    if skip_gdelt:
        df.to_csv(out_path, index=False, encoding="utf-8-sig"); return df

    # FAST: 상위 K만 GDELT
    if fast_mode:
        shortlist = df.sort_values(["naver_spike","naver_24h"], ascending=[False, False]).head(gdelt_top_k)
        target_names = set(shortlist["display"])
    else:
        target_names = set(df["display"])

    p2 = st.progress(0, text="2/2 GDELT 집계 중...")
    targets = [r for r in rows if r["display"] in target_names]
    total2 = len(targets)
    if total2 == 0:
        p2.empty()
        df = pd.DataFrame(rows)
        if min_unique_domains > 0:
            mask = df["unique_domains_24h"] >= min_unique_domains
            df.loc[~mask, ["combined_spike","naver_spike"]] = 0
        df.to_csv(out_path, index=False, encoding="utf-8-sig"); return df

    for jdx, r in enumerate(targets, start=1):
        g24 = gdelt_doc_count_smart(r["name_kr"], r["name_en"], "24h", cache_hours=gdelt_cache_h)
        g7  = gdelt_doc_count_smart(r["name_kr"], r["name_en"], "7d",  cache_hours=gdelt_cache_h)
        g30 = gdelt_doc_count_smart(r["name_kr"], r["name_en"], "30d", cache_hours=gdelt_cache_h)
        spike_g = spike_from_7d(g24, g7)
        r["gdelt_24h"], r["gdelt_7d"], r["gdelt_30d"], r["gdelt_spike"] = g24, g7, g30, round(spike_g,3)
        r["combined_spike"] = round(math.sqrt(max(r["naver_spike"],1e-4)*max(spike_g,1e-4)), 3) if (g24>0 or g7>0) else r["naver_spike"]
        r["newly_appeared"] = (r["naver_30d"] <= 1 and r["naver_24h"] >= 3) and (g30 <= 1)
        if jdx % 5 == 0: time.sleep(0.05)
        p2.progress(min(int(jdx/total2*100), 100), text=f"2/2 GDELT 집계 중... ({jdx}/{total2})")
    p2.empty()

    df = pd.DataFrame(rows)
    if min_unique_domains > 0:
        mask = df["unique_domains_24h"] >= min_unique_domains
        df.loc[~mask, ["combined_spike","naver_spike"]] = 0

    df.to_csv(out_path, index=False, encoding="utf-8-sig"); return df

# ─────────────── 표시/연산 안전가드 유틸 (NEW) ──────────────────────────
REQUIRED_NUMERIC = [
    "naver_spike","naver_24h","naver_7d","naver_30d",
    "gdelt_24h","gdelt_7d","gdelt_30d","combined_spike","unique_domains_24h",
    "naver_24h_weighted","naver_7d_weighted"
]
OPTIONAL_TEXT = ["display","ticker","name_kr","name_en","type","theme","aliases"]

def ensure_columns_and_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in OPTIONAL_TEXT:
        if c not in df.columns: df[c] = ""
        df[c] = df[c].astype(str).fillna("").str.strip()
    for c in REQUIRED_NUMERIC:
        if c not in df.columns: df[c] = 0
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if "newly_appeared" not in df.columns: df["newly_appeared"] = False
    else:
        df["newly_appeared"] = df["newly_appeared"].astype(bool)
    return df

def quick_health(df: pd.DataFrame) -> dict:
    need = ["display","ticker","combined_spike","naver_spike","naver_24h","naver_7d","gdelt_24h"]
    out = {"rows": len(df), "missing": [c for c in need if c not in df.columns]}
    na_rate = df.isna().mean().sort_values(ascending=False)
    out["na_top5"] = na_rate.head(5).round(3).to_dict()
    for c in ["combined_spike","naver_spike","naver_24h","naver_7d","gdelt_24h"]:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").fillna(0)
            out[f"{c}_neg"] = int((s < 0).sum()); out[f"{c}_zeros"] = int((s == 0).sum())
    return out
# ───────────────────────── 실시간 주가 (안정화) ─────────────────────────
def fetch_price_board(tickers: list[str]):
    if not yf or not tickers: return None
    rows = []; now_kst = datetime.now(KST).strftime("%m-%d %H:%M")
    for tk in tickers:
        price = chg = pct = None; ok = False
        trials = [("1mo","1d"), ("3mo","1d"), ("5d","60m"), ("5d","30m")]
        for period, interval in trials:
            try:
                hist = yf.Ticker(tk).history(period=period, interval=interval, auto_adjust=False)
                if hist is None or hist.empty: continue
                c = hist["Close"].dropna()
                if len(c)==0: continue
                last = float(c.iloc[-1]); prev = float(c.iloc[-2]) if len(c)>1 else last
                price = round(last,4); chg = round(last-prev,4); pct = round((chg/prev*100) if prev else 0.0, 2)
                ok = True; break
            except Exception: continue
        rows.append({"ticker": tk, "price": price, "change": chg, "pct": pct, "time": now_kst, "status": "OK" if ok else "EMPTY"})
    df = pd.DataFrame(rows)
    if (df["price"].isna()).all(): return None
    return df

# ====================== 뉴스×가격 신호/백테스트 유틸 ======================
@st.cache_data(show_spinner=False, ttl=2*60*60)
def build_news_daily_series(name_kr: str, name_en: str, display: str, days: int = 60,
                            bday_only: bool = True, holidays: set[date] | None = None) -> pd.Series:
    candidates = []
    if display: candidates.append(f"\"{display}\"")
    if name_en: candidates += [f"\"{name_en}\"", name_en]
    if name_kr: candidates += [f"\"{name_kr}\"", name_kr]
    for q in candidates:
        s = _gdelt_timeline_series(q, timespan=f"{days}d", cache_hours=6)
        if not s.empty:
            s = s.astype(int).sort_index()
            if bday_only:
                start = (datetime.now(KST) - timedelta(days=days)).date()
                end = datetime.now(KST).date()
                bidx = business_days_index(start, end, holidays or set())
                s = s.reindex(bidx, fill_value=0)
            else:
                s = s.asfreq("D").fillna(0)
            return s
    items = naver_news_search_all(display or name_kr or name_en, max_items=100, sort="date", cache_hours=2)
    counts = {}
    for it in items:
        pub = it.get("pubDate")
        try:
            dt = dtparser.parse(pub)
            if not dt.tzinfo: dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(KST).date()
            counts[str(dt)] = counts.get(str(dt), 0) + 1
        except Exception: continue
    if not counts: return pd.Series(dtype=int)
    s = pd.Series(counts); s.index = pd.to_datetime(s.index); s = s.sort_index()
    if bday_only:
        start = (datetime.now(KST) - timedelta(days=days)).date()
        end = datetime.now(KST).date()
        bidx = business_days_index(start, end, holidays or set())
        s = s.reindex(bidx, fill_value=0)
    else:
        s = s.asfreq("D").fillna(0)
    return s

def rolling_median_mad(s: pd.Series, win: int = 20):
    med = s.rolling(win, min_periods=max(2, win//2)).median()
    mad = s.rolling(win, min_periods=max(2, win//2)).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    mad = mad.replace(0, np.nan)
    return med, mad

def news_zscore(news_count: pd.Series, win: int = 20) -> pd.Series:
    news_count = news_count.astype(float).fillna(0)
    med, mad = rolling_median_mad(news_count, win)
    z = (news_count - med) / mad
    return z.replace([np.inf, -np.inf], np.nan)

def compute_indicators(px: pd.DataFrame) -> pd.DataFrame:
    df = px.copy()
    if not isinstance(df.index, pd.DatetimeIndex): df.index = pd.to_datetime(df.index)
    df["ret"] = df["Close"].pct_change()
    df["SMA10"] = df["Close"].rolling(10).mean()
    df["SMA20"] = df["Close"].rolling(20).mean()
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()
    if "Volume" in df: df["VOL_SPike"] = df["Volume"] / df["Volume"].rolling(20).mean()
    if "Value" in df:  df["VAL_SPike"] = df["Value"] / df["Value"].rolling(20).mean()
    return df

@dataclass
class SignalParams:
    z_win: int = 20
    z_th: float = 3.0
    vol_mult: float = 1.5
    atr_stop: float = 2.0
    atr_tp: float = 4.0
    use_value: bool = False
    min_unique_domains: int = 3  # 허수 방지 필터

def generate_signals(px_df: pd.DataFrame,
                     news_daily_counts: pd.Series,
                     params: SignalParams,
                     mode: str = "breakout",
                     unique_domains_24h: int | None = None) -> pd.DataFrame:
    df = compute_indicators(px_df)
    df["NEWS_CNT"] = news_daily_counts.reindex(df.index).fillna(0)
    df["NEWS_Z"] = news_zscore(df["NEWS_CNT"], params.z_win)

    # 허수 방지
    if unique_domains_24h is not None and unique_domains_24h < params.min_unique_domains:
        df["BUY"], df["SELL"], df["reason"] = False, False, "blocked_low_unique_domains"
        df["entry_price"] = np.nan; df["stop_price"] = np.nan; df["take_price"] = np.nan
        df["strategy"] = mode
        return df

    # Spike 시리즈 안전 생성
    spike = df["VAL_SPike"] if (params.use_value and "VAL_SPike" in df) else df.get("VOL_SPike")
    if spike is None:
        spike = pd.Series(0.0, index=df.index)
    else:
        spike = pd.to_numeric(spike, errors="coerce").fillna(0.0).astype(float)

    df["BUY"], df["SELL"], df["reason"] = False, False, ""

    if mode == "breakout":
        cond = ((df["NEWS_Z"] >= params.z_th) &
                (df["Close"] > df["Close"].rolling(20).max().shift(1)) &
                (spike >= params.vol_mult) &
                (df["Close"] >= df["SMA20"]))
        df.loc[cond, ["BUY","reason"]] = True, "news_z+breakout"

    elif mode == "pullback":
        y = df["NEWS_Z"].shift(1) >= params.z_th
        cond = y & (df["High"] > df["High"].shift(1)) & (df["Close"] >= df["SMA20"]) & (spike >= params.vol_mult)
        df.loc[cond, ["BUY","reason"]] = True, "news_pullback_break"

    elif mode == "drift":
        streak = (df["NEWS_Z"] > 0).rolling(3).sum() >= 3
        pos3 = (df["Close"].pct_change().rolling(3).sum() > 0)
        cond = streak & pos3 & (df["SMA10"] > df["SMA20"])
        df.loc[cond, ["BUY","reason"]] = True, "news_drift"

    df["entry_price"] = df["Open"].shift(-1).where(df["BUY"])  # 다음 캔들 시가
    df["stop_price"]  = (df["entry_price"] - params.atr_stop * df["ATR14"]).where(df["BUY"])
    df["take_price"]  = (df["entry_price"] + params.atr_tp   * df["ATR14"]).where(df["BUY"])
    df["strategy"] = mode
    return df

def backtest(df: pd.DataFrame, slippage_bps: float = 3.0, commission_bps: float = 3.0):
    slip = float(slippage_bps) / 1e4
    comm = float(commission_bps) / 1e4
    trades = []
    for dt, r in df[df["BUY"]].iterrows():
        e = r.get("entry_price", np.nan)
        if pd.isna(e) or e == 0: continue
        e_cost = e * (1 + slip + comm)
        stop_p = r.get("stop_price", np.nan); take_p = r.get("take_price", np.nan)
        try: idx = df.index.get_loc(dt)
        except KeyError: continue
        nxt = idx + 1; outcome, exit_p = None, None
        if nxt < len(df.index):
            lo1 = df.iloc[nxt]["Low"]; hi1 = df.iloc[nxt]["High"]
            if not pd.isna(stop_p) and lo1 <= stop_p: exit_p, outcome = stop_p, "stop"
            if outcome is None and (not pd.isna(take_p)) and hi1 >= take_p: exit_p, outcome = take_p, "take"
        if exit_p is None: exit_p, outcome = df.at[dt, "Close"], "eod"
        x_cost = exit_p * (1 - slip - comm)
        ret = (x_cost - e_cost) / e_cost
        trades.append({"date": dt, "entry": float(e_cost), "exit": float(x_cost), "ret": float(ret),
                       "outcome": outcome, "reason": r.get("reason",""), "strategy": r.get("strategy","")})
    if not trades:
        stats = {"trades":0, "win_rate":np.nan, "avg_ret":np.nan, "cagr":np.nan,
                 "mdd":np.nan, "sharpe":np.nan, "take_ratio":np.nan, "stop_ratio":np.nan}
        return pd.DataFrame(), stats
    tdf = pd.DataFrame(trades).set_index("date").sort_index()
    eq = (1 + tdf["ret"]).cumprod(); peak = eq.cummax(); dd = eq/peak - 1
    ann = max(1, int(len(eq)/252))
    sr = tdf["ret"].mean() / (tdf["ret"].std() + 1e-9) * np.sqrt(252) if tdf["ret"].std() > 0 else np.nan
    stats = {
        "trades": int(len(tdf)),
        "win_rate": float((tdf["ret"]>0).mean()),
        "avg_ret": float(tdf["ret"].mean()),
        "cagr": float(eq.iloc[-1]**ann - 1),
        "mdd": float(dd.min()),
        "sharpe": float(sr),
        "take_ratio": float((tdf["outcome"]=="take").mean()),
        "stop_ratio": float((tdf["outcome"]=="stop").mean()),
    }
    return tdf, stats

# ───────────────────────── UI (상단) ─────────────────────────
st.title("뉴스 언급 급등 — 통합 집계 & 대시보드 (NAVER × GDELT)")
st.caption("도메인 가중 스파이크 + 거래일 보정 Z + 허수 방지 + 슬리피지/수수료 + 프리셋")
st.sidebar.caption(f"🔄 마지막 갱신: {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S KST')}")

with st.sidebar:
    st.header("옵션")
    top_n = st.slider("Top N", 5, 50, 15, 1)
    max_entities = st.slider("이번 실행에서 처리할 최대 엔티티 수", 50, 3000, 300, 50)
    st.markdown("---")
    fast_mode = st.checkbox("FAST 모드 (상위 K만 GDELT)", value=True)
    gdelt_top_k = st.slider("FAST 모드에서 GDELT 계산할 K", 10, 500, 100, 10, disabled=not fast_mode)
    skip_gdelt = st.checkbox("GDELT 완전 건너뛰기 (NAVER만)", value=False)
    st.markdown("---")
    min_unique_domains = st.slider("정확도 필터: 24h 고유 도메인 최소값", 0, 10, 3, 1)
    st.caption("※ 3~5 권장. 허수 뉴스 억제")
    st.markdown("---")
    naver_cache_h = st.slider("NAVER 캐시 시간(시간)", 1, 24, 6, 1)
    gdelt_cache_h = st.slider("GDELT 캐시 시간(시간)", 1, 24, 6, 1)
    st.markdown("---")
    sort_choice = st.selectbox("정렬 기준(표/차트)", ["combined_spike","naver_spike","naver_24h","gdelt_24h"], index=0)
    st.markdown("---")
    st.write("📄 엔티티 미리보기 (entities.csv)")
    try:
        ents_df = pd.read_csv(ENTITIES_CSV, encoding="utf-8-sig")
    except Exception:
        try: ents_df = pd.read_csv(ENTITIES_CSV, encoding="cp949")
        except Exception: ents_df = None
    if ents_df is not None:
        st.dataframe(ents_df.head(200), use_container_width=True, hide_index=True)
    else:
        st.info("entities.csv 를 불러올 수 없습니다.")
    st.markdown("---")
    st.subheader("상세 보기 옵션")
    news_limit = st.slider("기사 요약 개수", 3, 20, 10, 1)
    enable_themes = st.checkbox("AI 테마 자동 분류 보기", value=True)
    enable_prices = st.checkbox("실시간 주가 보드 표시(티커 있는 종목만)", value=True)
    protect_naver = st.checkbox("Naver API 보호 모드(테마 개요 뉴스 호출 생략)", value=False)
    st.markdown("---")
    # 헬스체크
    try:
        if os.path.exists(OUT_CSV):
            _tmp = pd.read_csv(OUT_CSV)
            st.sidebar.json({"health": quick_health(_tmp)})
    except Exception:
        pass

# ‘처음 등장’ 조건(유연 조절)
with st.sidebar.expander("‘처음 등장’ 조건"):
    th_24h = st.slider("NAVER 24h ≥", 1, 10, 3, 1)
    th_30d = st.slider("NAVER 30d ≤", 0, 5, 1, 1)
    th_g30 = st.slider("GDELT 30d ≤", 0, 5, 1, 1)

st.write(f"**기준 시각(KST)**: {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')}")

# 집계 실행/로드
col1, col2 = st.columns([1,1])
with col1:
    if st.button("🚀 집계 실행 (CSV 생성/갱신)"):
        if not CLIENT_ID or not CLIENT_SECRET:
            st.error("`.env`의 NAVER_CLIENT_ID / NAVER_CLIENT_SECRET가 필요합니다.")
        else:
            with st.spinner("집계 중..."):
                entities = load_entities(ENTITIES_CSV)
                random.shuffle(entities); entities = entities[:max_entities]
                df = generate_cross_csv(
                    entities=entities, out_path=OUT_CSV,
                    fast_mode=fast_mode, gdelt_top_k=gdelt_top_k, skip_gdelt=skip_gdelt,
                    min_unique_domains=min_unique_domains, naver_cache_h=naver_cache_h, gdelt_cache_h=gdelt_cache_h,
                )
            st.success(f"CSV 저장 완료: {OUT_CSV} (행 {len(df)})"); st.session_state["df"] = df

with col2:
    if st.button("🔄 CSV 다시 불러오기"):
        if os.path.exists(OUT_CSV):
            st.session_state["df"] = pd.read_csv(OUT_CSV); st.success("CSV 로드 완료")
        else:
            st.warning(f"{OUT_CSV}가 없습니다. 먼저 집계를 실행하세요.")

# 데이터 소스
df_src = None
if "df" in st.session_state: df_src = st.session_state["df"]
elif os.path.exists(OUT_CSV): df_src = pd.read_csv(OUT_CSV)
if df_src is None or df_src.empty:
    st.info("표시할 데이터가 없습니다. 먼저 **집계 실행**을 눌러 CSV를 생성하세요."); st.stop()

# ── 안전가드 적용 ─────────────────────────────────────────────────────────
df_base = ensure_columns_and_types(df_src)

# ── 정렬/뷰 ────────────────────────────────────────────────────────────────
if sort_choice not in df_base.columns: df_base[sort_choice] = 0
df_sorted = df_base.sort_values([sort_choice,"naver_24h","gdelt_24h"], ascending=[False,False,False]).reset_index(drop=True)

st.subheader("교차 급등 TOP")
safe_cols = [c for c in [
    "display","ticker","combined_spike","naver_spike","naver_24h","naver_24h_weighted",
    "naver_7d","naver_7d_weighted","unique_domains_24h",
    "gdelt_24h","gdelt_7d","naver_30d","gdelt_30d","newly_appeared"
] if c in df_sorted.columns]
top_table = df_sorted.loc[:, safe_cols].head(top_n).copy()
for c in ["naver_24h_weighted","naver_7d_weighted"]:
    if c in top_table.columns:
        top_table[c] = top_table[c].replace(0, np.nan).fillna("–")
st.dataframe(top_table, use_container_width=True, hide_index=True)

st.subheader("상위 종목 막대차트 — " + sort_choice)
top = df_sorted.head(top_n).set_index("display")
st.bar_chart(top[sort_choice])

# Themes
if enable_themes:
    st.subheader("Themes Overview (AI 분류)")
    theme_acc = {}
    if not protect_naver:
        for disp in df_sorted.head(top_n)["display"].tolist():
            cards = extract_news_cards(disp, limit=5, cache_hours=2)
            texts = [c["title"] + " " + c["summary"] for c in cards]
            tms = classify_themes(texts, top_k=3)
            for t in tms: theme_acc[t] = theme_acc.get(t, 0) + 1
    if theme_acc:
        theme_df = pd.DataFrame([{"theme":k, "count":v} for k,v in sorted(theme_acc.items(), key=lambda x: x[1], reverse=True)])
        st.dataframe(theme_df, use_container_width=True, hide_index=True)
    else:
        st.write("테마 감지 결과 없음 (또는 보호 모드)")

# ── 상세 인스펙터 ────────────────────────────────────────────────────────────
st.subheader("상위 종목 상세 인스펙터")
left, right = st.columns([1,1])

with left:
    candidates = df_sorted.head(top_n)["display"].tolist()
    target_disp = st.selectbox("종목/기업 선택", candidates)
    row = df_sorted[df_sorted["display"]==target_disp].head(1)
    ticker = (row["ticker"].iloc[0] if "ticker" in row.columns and len(row)>0 else "") or ""
    st.markdown(f"**선택:** {target_disp}" + (f"  |  **티커:** `{ticker}`" if ticker else ""))

    st.markdown("##### 📰 최신 기사 요약/링크")
    try: cards = extract_news_cards(target_disp, limit=news_limit, cache_hours=2)
    except Exception: cards = []
    if cards:
        news_df = pd.DataFrame(cards)
        news_df["link"] = news_df["link"].fillna("")  # 안전가드
        news_df = news_df[["time","media","title","summary","link"]]
        st.dataframe(news_df, use_container_width=True, hide_index=True,
                     column_config={"link": st.column_config.LinkColumn("link")})
    else:
        st.write("표시할 뉴스가 없습니다.")

    if enable_themes and cards:
        texts = [c["title"] + " " + c["summary"] for c in cards]
        themes = classify_themes(texts, top_k=5)
        st.markdown("**AI 추정 테마:** " + (", ".join(themes) if themes else "(없음)"))

with right:
    if enable_prices:
        st.markdown("##### 💹 실시간 주가 보드")
        tickers_top = [t for t in df_sorted.head(top_n)["ticker"].fillna("").tolist() if t]
        tickers_all = [t for t in df_sorted["ticker"].fillna("").tolist() if t]
        base_tickers = tickers_top if tickers_top else tickers_all[:20]
        if ticker: base_tickers = list({*base_tickers, ticker})
        manual = st.text_input("수동 테스트 티커 추가 (예: AAPL, NVDA, 005930.KS)", value="")
        extra = [t.strip() for t in manual.split(",") if t.strip()]
        if extra: base_tickers = list({*base_tickers, *extra})
        if not yf:
            st.warning("yfinance가 설치되어 있지 않습니다. `pip install yfinance` 후 이용하세요.")
        elif not base_tickers:
            st.info("표시할 티커가 없습니다. CSV의 `ticker` 컬럼을 채우거나, 위 입력란에 테스트 티커를 추가하세요.")
        else:
            board = fetch_price_board(base_tickers)
            name_map = {}
            if "ticker" in df_sorted.columns:
                name_map = (df_sorted.loc[df_sorted["ticker"].fillna("")!="", ["ticker","display"]]
                                    .drop_duplicates(subset=["ticker"]).set_index("ticker")["display"].to_dict())
            if board is not None and not board.empty:
                board.insert(0, "name", board["ticker"].map(name_map).fillna(board["ticker"]))
                board = board[["name","ticker","price","change","pct","time","status"]].rename(columns={
                    "name":"회사명","ticker":"티커","price":"가격","change":"전일대비","pct":"등락률(%)","time":"시간","status":"상태"
                })
            if board is not None and not board.empty and board["가격"].notna().any():
                st.dataframe(board, use_container_width=True, hide_index=True)
            else:
                st.error("가격 데이터를 가져올 수 없습니다. (야후 포맷 .KS/.KQ, 방화벽 등 확인)")

        if yf and ticker:
            try:
                hist = yf.Ticker(ticker).history(period="1mo", interval="1d", auto_adjust=False)
                if hist is not None and not hist.empty:
                    disp_name = name_map.get(ticker, target_disp) if 'name_map' in locals() else target_disp
                    st.markdown(f"**{disp_name} ({ticker})** — 1M 종가")
                    st.line_chart(hist["Close"])
            except Exception:
                pass

# ── 매수 타이밍 — 3전략 통합 ────────────────────────────────────────────────
st.markdown("---")
st.header("📈 매수 타이밍 — 3전략 통합")

kr_holidays = load_kr_holidays(KR_HOLIDAY_CSV)

with st.sidebar.expander("신호 파라미터 (공통)", expanded=True):
    z_win = st.slider("뉴스 z 윈도우(영업일)", 10, 40, 20, 1, key="z_win")
    z_th  = st.slider("뉴스 z 임계치", 2.0, 6.0, 3.0, 0.1, key="z_th")
    vol_mult = st.slider("거래량/대금 배수", 1.0, 5.0, 1.5, 0.1, key="vol_mult")
    atr_stop = st.slider("손절 ATR배수", 1.0, 4.0, 2.0, 0.1, key="atr_stop")
    atr_tp   = st.slider("익절 ATR배수", 2.0, 8.0, 4.0, 0.1, key="atr_tp")
    use_value = st.checkbox("거래대금(Value) 사용", value=False, help="가격 데이터에 Value 컬럼 존재 시", key="use_value")
    min_ud = st.slider("허수 방지: 최소 고유 도메인(24h)", 0, 10, min_unique_domains, 1, key="min_ud")

with st.sidebar.expander("전략 가중치 + 프리셋", expanded=True):
    def set_preset(kind: str):
        if kind == "normal":
            st.session_state["w_break"]=0.15; st.session_state["w_pull"]=0.25; st.session_state["w_drift"]=0.60
        elif kind == "bull":
            st.session_state["w_break"]=0.50; st.session_state["w_pull"]=0.10; st.session_state["w_drift"]=0.40
        elif kind == "choppy":
            st.session_state["w_break"]=0.10; st.session_state["w_pull"]=0.40; st.session_state["w_drift"]=0.50
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("평시 프리셋"): set_preset("normal")
    with c2:
        if st.button("강세 프리셋"): set_preset("bull")
    with c3:
        if st.button("변동 프리셋"): set_preset("choppy")

    w_break = st.slider("돌파형 가중치", 0.0, 1.0, st.session_state.get("w_break", 0.15), 0.05, key="w_break")
    w_pull  = st.slider("되돌림형 가중치", 0.0, 1.0, st.session_state.get("w_pull", 0.25), 0.05, key="w_pull")
    w_drift = st.slider("드리프트형 가중치", 0.0, 1.0, st.session_state.get("w_drift", 0.60), 0.05, key="w_drift")

with st.sidebar.expander("거래 코스트(슬리피지·수수료)", expanded=True):
    slip_bps = st.number_input("슬리피지 (bps, 0.01% = 1bps)", min_value=0.0, max_value=50.0, value=3.0, step=0.5, key="slip_bps")
    comm_bps = st.number_input("왕복 수수료 (bps)", min_value=0.0, max_value=50.0, value=3.0, step=0.5, key="comm_bps")
    st.caption("권장 범위 2~5 bps (0.02%~0.05%)")

# 선택 종목 뉴스/가격
try:
    name_kr = row["name_kr"].iloc[0] if "name_kr" in row.columns else target_disp
    name_en = row["name_en"].iloc[0] if "name_en" in row.columns else ""
    news_series = build_news_daily_series(name_kr=name_kr, name_en=name_en, display=target_disp, days=60,
                                          bday_only=True, holidays=kr_holidays)
    px_df = None
    if yf and ticker:
        px = yf.Ticker(ticker).history(period="6mo", interval="1d", auto_adjust=False)
        if px is not None and not px.empty:
            px_df = px.copy()[["Open","High","Low","Close","Volume"]]
            px_df.index = pd.to_datetime(px_df.index)
except Exception:
    news_series = pd.Series(dtype=int); px_df = None

# ★★★ TZ 패치 적용: 인덱스 통일 후 reindex 진행
if px_df is None or px_df.empty:
    st.info("⚠️ 티커가 없는 종목입니다. 티커가 있는 종목을 선택하거나 수동으로 추가해 주세요.")
else:
    px_df.index = to_naive_kst(px_df.index)
    if isinstance(news_series, pd.Series) and not news_series.empty:
        news_series.index = to_naive_kst(news_series.index)

    params = SignalParams(z_win=z_win, z_th=z_th, vol_mult=vol_mult, atr_stop=atr_stop, atr_tp=atr_tp,
                          use_value=use_value, min_unique_domains=min_ud)
    ud24 = int(row["unique_domains_24h"].iloc[0]) if "unique_domains_24h" in row.columns else None

    out = {}
    for mode in ["breakout","pullback","drift"]:
        sig_df = generate_signals(px_df, news_series, params, mode, unique_domains_24h=ud24)
        trades, stats = backtest(sig_df, slippage_bps=slip_bps, commission_bps=comm_bps)
        out[mode] = {"sig": sig_df, "trades": trades, "stats": stats}

    st.subheader(f"신호 성과 요약 — {target_disp}" + (f" ({ticker})" if ticker else ""))

    def fmt_pct(x): return f"{x*100:.1f}%" if pd.notna(x) else "—"
    def fmt_float(x, n=2): return f"{x:.{n}f}" if pd.notna(x) else "—"

    cols = st.columns(4)
    for i, mode in enumerate(["breakout","pullback","drift"]):
        stats = out[mode]["stats"]
        with cols[i]:
            st.metric(f"{mode}", f"승률 {fmt_pct(stats['win_rate'])}",
                      help=f"거래수 {stats['trades']} · Sharpe {fmt_float(stats['sharpe'])} · MDD {fmt_pct(stats['mdd'])}")

    def score(stats):
        if any(pd.isna(stats.get(k)) for k in ["win_rate","sharpe","mdd"]): return np.nan
        denom = 1 + abs(stats["mdd"]); return (stats["win_rate"] * max(stats["sharpe"], 0)) / denom

    sb, sp, sd = score(out["breakout"]["stats"]), score(out["pullback"]["stats"]), score(out["drift"]["stats"])
    blended = (st.session_state["w_break"]*(sb if pd.notna(sb) else 0) +
               st.session_state["w_pull"] *(sp if pd.notna(sp) else 0) +
               st.session_state["w_drift"]*(sd if pd.notna(sd) else 0))
    st.caption(f"가중 통합 점수(참고치): {fmt_float(blended, 3)}  ·  가중치 (돌파 {st.session_state['w_break']:.2f} / 되돌림 {st.session_state['w_pull']:.2f} / 드리프트 {st.session_state['w_drift']:.2f})")

    st.markdown("#### 뉴스 시계열 / 뉴스 Z / 매수 신호")
    if isinstance(news_series, pd.Series) and not news_series.empty:
        nz = news_zscore(news_series, z_win).reindex(px_df.index, method="ffill")
        chart_df = pd.DataFrame({"NEWS_CNT": news_series.reindex(px_df.index).fillna(0), "NEWS_Z": nz})
        st.line_chart(chart_df)

    st.markdown("#### 트레이드 로그")
    tabs = st.tabs(["돌파형", "되돌림형", "드리프트형"])
    for tab, mode in zip(tabs, ["breakout","pullback","drift"]):
        with tab:
            trades = out[mode]["trades"]; stats = out[mode]["stats"]
            st.write(f"거래수 {stats['trades']} · 승률 {fmt_pct(stats['win_rate'])} · 평균수익 {fmt_pct(stats['avg_ret'])} · Sharpe {fmt_float(stats['sharpe'])} · MDD {fmt_pct(stats['mdd'])}")
            if trades is not None and not trades.empty:
                st.dataframe(trades.tail(50), use_container_width=True)
            else:
                block_msg = " (고유 도메인 부족으로 차단됨)" if ud24 is not None and ud24 < min_ud else ""
                st.info("신호가 없거나 체결된 트레이드가 없습니다" + block_msg + ". 파라미터를 조정해 보세요.")

    st.markdown("#### 최근 신호 시각화(간단)")
    try:
        last_mode = st.selectbox("전략 선택", ["breakout","pullback","drift"], index=0)
        sig_df = out[last_mode]["sig"].copy()
        viz = pd.DataFrame({"Close": px_df["Close"]}); viz["BUY"] = sig_df["BUY"].astype(int)
        st.line_chart(viz[["Close","BUY"]])
    except Exception:
        pass

# ── ‘처음 등장’ 후보 (df_base 기준 계산) ────────────────────────────────────
st.subheader("‘처음 등장’ 후보")
tmp = df_base.copy()
for c in ["naver_24h","naver_30d","gdelt_30d","gdelt_24h"]:
    if c not in tmp.columns: tmp[c] = 0
    tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0)

new_df = tmp[(tmp["naver_30d"] <= th_30d) & (tmp["naver_24h"] >= th_24h) & (tmp["gdelt_30d"] <= th_g30)]
new_df = new_df.sort_values(["naver_24h","gdelt_30d","naver_30d"], ascending=[False, True, True])

if new_df.empty:
    st.write(f"⚪ 현재 조건(NAVER 30d≤{th_30d} & 24h≥{th_24h} AND GDELT 30d≤{th_g30})에 해당 없음")
else:
    st.dataframe(
        new_df.loc[:, ["display","ticker","naver_24h","naver_30d","gdelt_24h","gdelt_30d","combined_spike"]].head(100),
        use_container_width=True, hide_index=True
    )
