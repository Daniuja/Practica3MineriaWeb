#!/usr/bin/env python3
"""
Practica 3 - Mineria de Uso de la Web.

Analisis reproducible con pandas y matplotlib.
Genera:
- tablas CSV
- graficos PNG
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

try:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Faltan librerias de analisis. Instala las dependencias con:\n"
        "python -m pip install -r requirements.txt"
    ) from exc


LOG_PATH = ROOT / "NASA_access_log_FULL.txt"
OUT = ROOT / "resultados_practica3"
TABLES = OUT / "tablas"
FIGS = OUT / "graficos"

REFERENCE = pd.Timestamp("1995-01-01", tz="UTC")
TIMEOUT_SECONDS = 30 * 60
AUTO_THRESHOLD = 0.5
ALLOWED_EXTS = {".htm", ".html", ".pdf", ".asp", ".exe", ".txt", ".doc", ".ppt", ".xls", ".xml", ""}
BOT_WORDS = [
    "webcrawler",
    "crawler",
    "spider",
    "robot",
    "bot",
    "slurp",
    "scooter",
    "lycos",
    "infoseek",
    "inktomi",
    "harvest",
    "architext",
    "worm",
]

LOG_RE = re.compile(
    r'^(?P<host>\S+) (?P<password>\S+) (?P<user>\S+) \[(?P<datetime>[^\]]+)\] '
    r'"(?P<request>[^"]*)" (?P<status>\S+) (?P<size>\S+)'
)


def ensure_dirs() -> None:
    OUT.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)


def parse_request(request: str) -> tuple[str, str, str]:
    parts = request.split()
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], ""
    return "", parts[0] if parts else "", ""


def clean_page(page: str) -> str:
    page = page.split("?", 1)[0].split("#", 1)[0]
    return page or "/"


def get_extension(page: str) -> str:
    filename = clean_page(page).rsplit("/", 1)[-1]
    if "." not in filename:
        return ""
    ext = "." + filename.rsplit(".", 1)[-1].lower()
    return ext if len(ext) <= 12 else ""


def get_directory(page: str) -> str:
    page = clean_page(page)
    return "/" if "/" not in page.strip("/") else page.rsplit("/", 1)[0] + "/"


def base_domain(host: str) -> str:
    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host):
        return host
    parts = host.lower().split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host.lower()


def domain_type(host: str) -> str:
    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host):
        return "[ip]"
    parts = host.lower().split(".")
    return "." + parts[-1] if len(parts) > 1 else "[sin dominio]"


def bot_category(host: str) -> str | None:
    host = host.lower()
    return next((word for word in BOT_WORDS if word in host), None)


def load_log() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    rows = []
    ext_counter = {}
    bot_counter = {}
    bot_host_counter = {}
    info = {"raw_records": 0, "parse_errors": 0, "filtered_records": 0, "missing_size": 0}

    with LOG_PATH.open("r", encoding="latin-1", errors="replace") as file:
        for line in file:
            info["raw_records"] += 1
            match = LOG_RE.match(line)
            if not match:
                info["parse_errors"] += 1
                continue

            data = match.groupdict()
            method, page, protocol = parse_request(data["request"])
            page = clean_page(page)
            ext = get_extension(page)
            ext_counter[ext or "[sin extension]"] = ext_counter.get(ext or "[sin extension]", 0) + 1

            if ext not in ALLOWED_EXTS:
                continue
            info["filtered_records"] += 1

            bot = bot_category(data["host"])
            if bot:
                bot_counter[bot] = bot_counter.get(bot, 0) + 1
                bot_host_counter[data["host"]] = bot_host_counter.get(data["host"], 0) + 1
                continue

            size = np.nan if data["size"] == "-" else int(data["size"])
            info["missing_size"] += int(pd.isna(size))
            rows.append(
                {
                    "host": data["host"],
                    "password": data["password"],
                    "user": data["user"],
                    "datetime": data["datetime"],
                    "method": method,
                    "page": page,
                    "protocol": protocol,
                    "status": data["status"],
                    "size": size,
                    "ext": ext,
                }
            )

    df = pd.DataFrame(rows)
    df["usuario_id"] = df["host"]
    df["date"] = pd.to_datetime(df["datetime"], format="%d/%b/%Y:%H:%M:%S %z", utc=True)
    df["timestamp"] = (df["date"] - REFERENCE).dt.total_seconds().astype("int64")
    df["hour"] = df["date"].dt.hour
    df["domain"] = df["host"].map(base_domain)
    df["domain_type"] = df["host"].map(domain_type)
    df["directory"] = df["page"].map(get_directory)

    extensions = pd.Series(ext_counter, name="accesos").sort_values(ascending=False)
    bots = pd.DataFrame({"categoria": bot_counter.keys(), "registros": bot_counter.values()})
    if not bots.empty:
        bots = bots.sort_values("registros", ascending=False)
        bots["proporcion_sobre_automaticos"] = bots["registros"] / bots["registros"].sum()

    bot_hosts = pd.DataFrame({"host": bot_host_counter.keys(), "registros": bot_host_counter.values()})
    if not bot_hosts.empty:
        bot_hosts = bot_hosts.sort_values("registros", ascending=False)
        bot_hosts["proporcion_sobre_automaticos"] = bot_hosts["registros"] / bot_hosts["registros"].sum()

    return df, extensions, bots, bot_hosts, info


def add_sessions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["usuario_id", "timestamp"]).copy()
    gap = df.groupby("usuario_id")["timestamp"].diff().fillna(TIMEOUT_SECONDS + 1)
    df["new_session"] = gap > TIMEOUT_SECONDS
    df["session_id"] = df["new_session"].cumsum().astype("int64")
    return df.drop(columns="new_session")


def build_sessions(df: pd.DataFrame) -> pd.DataFrame:
    sessions = (
        df.groupby("session_id")
        .agg(
            host=("host", "first"),
            usuario_id=("usuario_id", "first"),
            start=("timestamp", "min"),
            end=("timestamp", "max"),
            visitas_pagina=("page", "size"),
            entrada=("page", "first"),
            salida=("page", "last"),
            hora_inicio=("hour", "first"),
        )
        .reset_index()
    )
    sessions["duracion_s"] = sessions["end"] - sessions["start"]
    sessions["tiempo_medio_pagina_s"] = np.where(
        sessions["visitas_pagina"] > 1,
        sessions["duracion_s"] / (sessions["visitas_pagina"] - 1),
        0,
    )
    return sessions


def save_table(name: str, data: pd.DataFrame | pd.Series) -> pd.DataFrame:
    df = data.reset_index() if isinstance(data, pd.Series) else data.copy()
    df.to_csv(TABLES / name, index=False, encoding="utf-8")
    return df


def stats_table(items: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, values in items.items():
        values = pd.to_numeric(values, errors="coerce").dropna()
        rows.append(
            {
                "medida": name,
                "n": len(values),
                "media": values.mean(),
                "desviacion": values.std(ddof=0),
                "mediana": values.median(),
                "moda": values.mode().iloc[0] if not values.mode().empty else np.nan,
                "minimo": values.min(),
                "maximo": values.max(),
            }
        )
    return pd.DataFrame(rows)


def capped(values: pd.Series, q: float = 0.99) -> tuple[pd.Series, float, int]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    cap = values.quantile(q)
    omitted = int((values > cap).sum())
    return values[values <= cap], float(cap), omitted


def hist(values: pd.Series, title: str, xlabel: str, filename: str) -> None:
    visible, cap, omitted = capped(values)
    total = len(pd.to_numeric(values, errors="coerce").dropna())
    omitted_pct = omitted / total * 100 if total else 0
    plt.figure(figsize=(9, 5))
    plt.hist(visible, bins=35, color="#2f7f7b", edgecolor="white")
    plt.title(f"{title}\nPercentil 99 = {cap:.2f}; omitidos = {omitted} ({omitted_pct:.2f}%)")
    plt.xlabel(xlabel)
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(FIGS / filename, dpi=150)
    plt.close()


def bar(values: pd.Series, title: str, xlabel: str, ylabel: str, filename: str) -> None:
    plt.figure(figsize=(9, 5))
    values.plot(kind="bar", color="#2f7f7b")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(FIGS / filename, dpi=150)
    plt.close()


def scatter_regression(sessions: pd.DataFrame) -> tuple[float, float]:
    x = sessions["visitas_pagina"].astype(float)
    y = sessions["duracion_s"].astype(float)
    slope, intercept = np.polyfit(x, y, 1)

    x_cap = x.quantile(0.99)
    y_cap = y.quantile(0.99)
    shown = sessions[(sessions["visitas_pagina"] <= x_cap) & (sessions["duracion_s"] <= y_cap)]
    omitted = len(sessions) - len(shown)
    omitted_pct = omitted / len(sessions) * 100 if len(sessions) else 0

    plt.figure(figsize=(9, 5))
    plt.scatter(shown["visitas_pagina"], shown["duracion_s"], s=6, alpha=0.25)
    xs = np.array([shown["visitas_pagina"].min(), shown["visitas_pagina"].max()])
    plt.plot(xs, intercept + slope * xs, color="#c43d2b", linewidth=2)
    plt.title(
        "Relacion entre paginas visitadas y duracion\n"
        f"Percentil 99: paginas={x_cap:.2f}, duracion={y_cap:.2f}; omitidos={omitted} ({omitted_pct:.2f}%)"
    )
    plt.xlabel("Paginas por sesion")
    plt.ylabel("Duracion de sesion (s)")
    plt.tight_layout()
    plt.savefig(FIGS / "dispersion_visitas_duracion.png", dpi=150)
    plt.close()
    return float(slope), float(intercept)


def first_page_durations(df: pd.DataFrame, human_ids: set[int]) -> pd.DataFrame:
    ordered = df[df["session_id"].isin(human_ids)].sort_values(["session_id", "timestamp"]).copy()
    ordered["position"] = ordered.groupby("session_id").cumcount() + 1
    ordered["next_timestamp"] = ordered.groupby("session_id")["timestamp"].shift(-1)
    ordered["page_duration_s"] = ordered["next_timestamp"] - ordered["timestamp"]
    return ordered[ordered["position"].isin([1, 2]) & ordered["page_duration_s"].notna()].copy()


def make_extra_tables(df: pd.DataFrame, sessions: pd.DataFrame, extensions: pd.Series) -> dict[str, pd.DataFrame]:
    tables = {}
    human_ids = set(sessions["session_id"])
    df = df[df["session_id"].isin(human_ids)]

    tables["07_top_20_dominios.csv"] = (
        pd.DataFrame(
            {
                "visitas": df.groupby("domain")["session_id"].nunique(),
                "clics_o_vistas": df.groupby("domain")["page"].size(),
            }
        )
        .fillna(0)
        .sort_values(["visitas", "clics_o_vistas"], ascending=False)
        .head(20)
        .reset_index()
    )
    tables["08_top_7_tipos_dominio.csv"] = (
        pd.DataFrame(
            {
                "visitas": df.groupby("domain_type")["session_id"].nunique(),
                "clics_o_vistas": df.groupby("domain_type")["page"].size(),
            }
        )
        .fillna(0)
        .sort_values(["visitas", "clics_o_vistas"], ascending=False)
        .head(7)
        .reset_index()
    )
    tables["09_longitud_media_visitas_24h.csv"] = sessions.groupby("hora_inicio", as_index=False).agg(
        visitas=("session_id", "size"),
        longitud_media_paginas=("visitas_pagina", "mean"),
        duracion_media_s=("duracion_s", "mean"),
    )
    tables["10_top_10_visitantes.csv"] = sessions["usuario_id"].value_counts().head(10).rename_axis("visitante").reset_index(name="visitas")
    visit_counts = sessions["usuario_id"].value_counts().value_counts()
    tables["11_visitantes_unicos_por_numero_visitas_1_9.csv"] = pd.DataFrame(
        {"numero_visitas": range(1, 10), "visitantes_unicos": [int(visit_counts.get(i, 0)) for i in range(1, 10)]}
    )
    tables["12_top_10_paginas.csv"] = (
        pd.DataFrame({"visitas": df.groupby("page")["session_id"].nunique(), "clics_o_vistas": df["page"].value_counts()})
        .fillna(0)
        .sort_values(["visitas", "clics_o_vistas"], ascending=False)
        .head(10)
        .reset_index(names="pagina")
    )
    tables["13_top_10_directorios.csv"] = (
        pd.DataFrame({"visitas": df.groupby("directory")["session_id"].nunique(), "clics_o_vistas": df["directory"].value_counts()})
        .fillna(0)
        .sort_values(["visitas", "clics_o_vistas"], ascending=False)
        .head(10)
        .reset_index(names="directorio")
    )
    tables["14_top_10_tipos_fichero.csv"] = extensions.head(10).rename_axis("extension").reset_index(name="accesos")
    tables["15_top_10_paginas_entrada.csv"] = sessions["entrada"].value_counts().head(10).rename_axis("pagina").reset_index(name="visitas")
    tables["16_top_10_paginas_salida.csv"] = sessions["salida"].value_counts().head(10).rename_axis("pagina").reset_index(name="visitas")
    single = sessions[sessions["visitas_pagina"] == 1]
    tables["17_top_10_paginas_acceso_unico.csv"] = single["entrada"].value_counts().head(10).rename_axis("pagina").reset_index(name="visitas")
    minute_counts = (sessions["duracion_s"] // 60).clip(upper=60).astype(int).value_counts().sort_index()
    minute_table = minute_counts.rename_axis("minuto_inicio").reset_index(name="visitas")
    minute_table["intervalo_minutos"] = minute_table["minuto_inicio"].map(
        lambda minute: "60+ min" if minute == 60 else f"{minute}-{minute + 1} min"
    )
    tables["18_duracion_visitas_minutos.csv"] = minute_table[["intervalo_minutos", "minuto_inicio", "visitas"]]
    return tables


def run_analysis() -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    ensure_dirs()
    df, extensions, bots, bot_hosts, info = load_log()
    df = add_sessions(df)
    sessions = build_sessions(df)

    auto_ids = set(sessions.query("visitas_pagina > 1 and tiempo_medio_pagina_s < @AUTO_THRESHOLD")["session_id"])
    human_sessions = sessions[~sessions["session_id"].isin(auto_ids)].copy()
    multi = human_sessions[human_sessions["visitas_pagina"] > 1].copy()
    human_ids = set(human_sessions["session_id"])

    tables = {
        "01_top_10_extensiones_original.csv": extensions.head(10).rename_axis("extension").reset_index(name="accesos"),
        "02_bots_y_crawlers.csv": bots,
        "03_hosts_bot_mas_frecuentes.csv": bot_hosts,
        "04_muestra_sesiones_ordenadas.csv": df[["session_id", "timestamp", "usuario_id", "host", "page", "ext"]].head(300),
        "05_20_sesiones_menor_tiempo_medio.csv": sessions.query("visitas_pagina > 1").nsmallest(20, "tiempo_medio_pagina_s"),
        "06_resumen_estadistico.csv": stats_table(
            {
                "duracion_sesion_s": multi["duracion_s"],
                "tiempo_medio_pagina_s": multi["tiempo_medio_pagina_s"],
                "paginas_por_sesion": human_sessions["visitas_pagina"],
            }
        ),
    }
    tables.update(make_extra_tables(df, human_sessions, extensions))

    first_second = first_page_durations(df, human_ids)
    first = first_second[first_second["position"] == 1]
    second = first_second[first_second["position"] == 2]
    tables["19_estadisticos_dos_primeras_paginas.csv"] = stats_table(
        {"primera_pagina_s": first["page_duration_s"], "segunda_pagina_s": second["page_duration_s"]}
    )
    first_second["tipo"] = np.where(first_second["ext"].eq(""), "navegacion", "contenido")
    tables["20_navegacion_vs_contenido.csv"] = (
        first_second.groupby(["position", "tipo"])["page_duration_s"]
        .agg(n="size", media_s="mean", mediana_s="median")
        .reset_index()
    )
    tables["21_muestra_registros_preprocesados.csv"] = df[df["session_id"].isin(human_ids)].head(1000)

    for name, table in tables.items():
        save_table(name, table)

    hist(multi["duracion_s"], "Histograma de duracion de sesion", "Segundos", "hist_duracion_sesion.png")
    hist(multi["tiempo_medio_pagina_s"], "Histograma de tiempo medio por pagina", "Segundos", "hist_tiempo_medio_pagina.png")
    hist(human_sessions["visitas_pagina"], "Paginas por sesion", "Paginas", "hist_paginas_por_sesion.png")
    hist(first["page_duration_s"], "Duracion de la primera pagina", "Segundos", "hist_duracion_primera_pagina.png")
    bar(
        tables["09_longitud_media_visitas_24h.csv"].set_index("hora_inicio")["longitud_media_paginas"],
        "Longitud media de las visitas por hora",
        "Hora",
        "Paginas medias por visita",
        "barras_longitud_media_24h.png",
    )
    slope, intercept = scatter_regression(multi)

    for position, filename in [(1, "hist_normalizado_nav_contenido_primera.png"), (2, "hist_normalizado_nav_contenido_segunda.png")]:
        subset = first_second[first_second["position"] == position]
        visible, cap, omitted = capped(subset["page_duration_s"])
        total = len(pd.to_numeric(subset["page_duration_s"], errors="coerce").dropna())
        omitted_pct = omitted / total * 100 if total else 0
        subset = subset[subset["page_duration_s"] <= cap]
        plt.figure(figsize=(9, 5))
        for tipo, group in subset.groupby("tipo"):
            plt.hist(group["page_duration_s"], bins=30, density=True, alpha=0.55, label=tipo)
        plt.title(f"Pagina {position}: navegacion vs contenido\nPercentil 99 = {cap:.2f}; omitidos = {omitted} ({omitted_pct:.2f}%)")
        plt.xlabel("Segundos")
        plt.ylabel("Frecuencia normalizada")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGS / filename, dpi=150)
        plt.close()

    context = {
        **info,
        "kept_records": len(df),
        "auto_sessions": len(auto_ids),
        "human_sessions": len(human_sessions),
        "slope": slope,
        "intercept": intercept,
    }
    return context, tables


def main() -> None:
    context, tables = run_analysis()
    print("Analisis terminado.")
    print(f"Registros procesados: {context['raw_records']:,}")
    print(f"Sesiones humanas finales: {context['human_sessions']:,}")
    print(f"Tablas generadas: {len(tables)} en {TABLES}")
    print(f"Graficos generados en {FIGS}")


if __name__ == "__main__":
    main()
