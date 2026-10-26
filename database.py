"""SQLite Persistence Layer for RoadZone AI.

Stores scanned road telemetry, geographic zones, pothole annotations,
Gemma 4 flood assessments, and citizen grievance reports.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path(__file__).parent / "roadzone.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables for zones, scans, incident reports, and hotspots."""
    with get_connection() as conn:
        cursor = conn.cursor()

        # Zones table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS zones (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                damage_bias TEXT,
                pothole_count INTEGER DEFAULT 0,
                avg_area_ratio REAL DEFAULT 0.0,
                pothole_score REAL DEFAULT 0.0,
                flood_severity REAL DEFAULT 0.0,
                final_score REAL DEFAULT 0.0,
                color TEXT DEFAULT 'green',
                rank INTEGER DEFAULT 0,
                updated_at TEXT
            )
            """
        )

        # Scans table: every image uploaded/analyzed
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                zone_id TEXT NOT NULL,
                image_name TEXT,
                pothole_count INTEGER DEFAULT 0,
                area_ratio REAL DEFAULT 0.0,
                is_flooded INTEGER DEFAULT 0,
                flood_severity REAL DEFAULT 0.0,
                calculated_score REAL DEFAULT 0.0,
                model_used TEXT DEFAULT 'Google Gemma 4 Vision',
                created_at TEXT,
                FOREIGN KEY (zone_id) REFERENCES zones(id)
            )
            """
        )

        # Incident reports table (Citizen & Telemetry Grievances)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS incident_reports (
                id TEXT PRIMARY KEY,
                zone_id TEXT NOT NULL,
                type TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT NOT NULL,
                reporter TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                status TEXT DEFAULT 'UNRESOLVED',
                FOREIGN KEY (zone_id) REFERENCES zones(id)
            )
            """
        )

        # Hotspots table (Specific GPS coordinates for craters and flood choke points)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS hotspots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                zone_id TEXT NOT NULL,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                severity TEXT,
                detected_by TEXT,
                area_damage TEXT,
                FOREIGN KEY (zone_id) REFERENCES zones(id)
            )
            """
        )

        conn.commit()


def save_or_update_zone(zone_dict: Dict[str, Any]):
    """Insert or update a zone record."""
    with get_connection() as conn:
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute(
            """
            INSERT INTO zones (
                id, name, description, lat, lon, damage_bias,
                pothole_count, avg_area_ratio, pothole_score, flood_severity,
                final_score, color, rank, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                pothole_count = excluded.pothole_count,
                avg_area_ratio = excluded.avg_area_ratio,
                pothole_score = excluded.pothole_score,
                flood_severity = excluded.flood_severity,
                final_score = excluded.final_score,
                color = excluded.color,
                rank = excluded.rank,
                updated_at = excluded.updated_at
            """,
            (
                zone_dict["id"],
                zone_dict["name"],
                zone_dict.get("description", ""),
                zone_dict["lat"],
                zone_dict["lon"],
                zone_dict.get("damage_bias", "clean"),
                zone_dict.get("pothole_count", 0),
                zone_dict.get("avg_area_ratio", 0.0),
                zone_dict.get("pothole_score", 0.0),
                zone_dict.get("flood_severity", 0.0),
                zone_dict.get("final_score", 0.0),
                zone_dict.get("color", "green"),
                zone_dict.get("rank", 0),
                now,
            ),
        )
        conn.commit()


def get_all_zones() -> List[Dict[str, Any]]:
    """Retrieve all zones sorted by rank."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM zones ORDER BY rank ASC")
        return [dict(row) for row in cursor.fetchall()]


def get_zone_by_id(zone_id: str) -> Optional[Dict[str, Any]]:
    """Get a single zone by its ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM zones WHERE id = ?", (zone_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def record_scan_in_db(
    zone_id: str,
    image_name: str,
    pothole_count: int,
    area_ratio: float,
    is_flooded: bool,
    flood_severity: float,
) -> Dict[str, Any]:
    """Save a new road scan and automatically recalculate the target zone's live score.
    
    Formula:
        Score = pothole_count + (10 * area_ratio) + (1.5 * flood_severity)
    """
    calc_score = round(pothole_count + (10.0 * area_ratio) + (1.5 * flood_severity), 2)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with get_connection() as conn:
        cursor = conn.cursor()

        # Insert scan
        cursor.execute(
            """
            INSERT INTO scans (
                zone_id, image_name, pothole_count, area_ratio,
                is_flooded, flood_severity, calculated_score, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                zone_id,
                image_name,
                pothole_count,
                area_ratio,
                1 if is_flooded else 0,
                flood_severity,
                calc_score,
                now,
            ),
        )

        # Aggregate all scans for this zone to update zone metrics
        cursor.execute(
            """
            SELECT 
                COUNT(*) as scan_count,
                SUM(pothole_count) as tot_potholes,
                AVG(area_ratio) as avg_area,
                AVG(flood_severity) as avg_flood
            FROM scans WHERE zone_id = ?
            """,
            (zone_id,),
        )
        agg = cursor.fetchone()
        scan_cnt = agg["scan_count"] or 1
        tot_p = agg["tot_potholes"] or pothole_count
        avg_ar = agg["avg_area"] or area_ratio
        avg_fl = agg["avg_flood"] or flood_severity

        new_pothole_score = round(tot_p + (10.0 * avg_ar), 2)
        new_final_score = round(new_pothole_score + (1.5 * avg_fl), 2)

        # Update zone table
        cursor.execute(
            """
            UPDATE zones
            SET pothole_count = ?,
                avg_area_ratio = ?,
                pothole_score = ?,
                flood_severity = ?,
                final_score = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (tot_p, avg_ar, new_pothole_score, avg_fl, new_final_score, now, zone_id),
        )

        conn.commit()

    # Recalculate all zone ranks and color quantiles
    _recalculate_ranks_and_colors()

    return {
        "zone_id": zone_id,
        "scan_score": calc_score,
        "new_zone_final_score": new_final_score,
        "potholes": tot_p,
        "flood_severity": avg_fl,
    }


def _recalculate_ranks_and_colors():
    """Recalculate ranks and Red/Orange/Green quantiles across all zones in DB."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, final_score FROM zones ORDER BY final_score DESC")
        rows = cursor.fetchall()
        n = len(rows)

        for rank_idx, r in enumerate(rows):
            rank = rank_idx + 1
            if rank_idx < n // 3:
                color = "red"
            elif rank_idx < (2 * n) // 3:
                color = "orange"
            else:
                color = "green"

            cursor.execute(
                "UPDATE zones SET rank = ?, color = ? WHERE id = ?",
                (rank, color, r["id"]),
            )
        conn.commit()


def get_all_scans(zone_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve historical scans."""
    with get_connection() as conn:
        cursor = conn.cursor()
        if zone_id:
            cursor.execute("SELECT * FROM scans WHERE zone_id = ? ORDER BY id DESC LIMIT ?", (zone_id, limit))
        else:
            cursor.execute("SELECT * FROM scans ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(row) for row in cursor.fetchall()]


def add_incident_report(report_dict: Dict[str, Any]):
    """Insert a citizen or telemetry incident report."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO incident_reports (
                id, zone_id, type, severity, description, reporter, timestamp, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                description = excluded.description,
                status = excluded.status
            """,
            (
                report_dict["id"],
                report_dict["zone_id"],
                report_dict["type"],
                report_dict["severity"],
                report_dict["description"],
                report_dict["reporter"],
                report_dict.get("timestamp", datetime.now().strftime("%B %d, %H:%M")),
                report_dict.get("status", "UNRESOLVED - ACTIVE DANGER"),
            ),
        )
        conn.commit()


def get_incident_reports(zone_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve incident reports."""
    with get_connection() as conn:
        cursor = conn.cursor()
        if zone_id:
            cursor.execute("SELECT * FROM incident_reports WHERE zone_id = ? ORDER BY id DESC", (zone_id,))
        else:
            cursor.execute("SELECT * FROM incident_reports ORDER BY id DESC")
        return [dict(row) for row in cursor.fetchall()]


def save_hotspot(hotspot_dict: Dict[str, Any]):
    """Insert a specific hotspot waypoint."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO hotspots (
                zone_id, type, title, lat, lon, severity, detected_by, area_damage
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                hotspot_dict["zone_id"],
                hotspot_dict["type"],
                hotspot_dict["title"],
                hotspot_dict["lat"],
                hotspot_dict["lon"],
                hotspot_dict.get("severity", ""),
                hotspot_dict.get("detected_by", ""),
                hotspot_dict.get("area_damage", ""),
            ),
        )
        conn.commit()


def get_hotspots_by_zone(zone_id: str) -> List[Dict[str, Any]]:
    """Get all specific hotspot waypoints for a zone."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM hotspots WHERE zone_id = ?", (zone_id,))
        return [dict(row) for row in cursor.fetchall()]


# Auto-initialize DB on import
init_db()


if __name__ == "__main__":
    print("Testing database.py...")
    init_db()
    print("Database tables initialized at:", DB_PATH)
