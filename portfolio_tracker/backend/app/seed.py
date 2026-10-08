"""Seed a starter set of frequently-tracked politicians.

Names must match the disclosure sources so scraped filings link to these rows.
`followed=True` here means the copy engine will consider their trades.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Chamber, Politician

SEED_POLITICIANS = [
    ("Nancy Pelosi", Chamber.HOUSE, "D", "CA", True),
    ("Dan Crenshaw", Chamber.HOUSE, "R", "TX", False),
    ("Josh Gottheimer", Chamber.HOUSE, "D", "NJ", False),
    ("Marjorie Taylor Greene", Chamber.HOUSE, "R", "GA", False),
    ("Michael McCaul", Chamber.HOUSE, "R", "TX", False),
    ("Ro Khanna", Chamber.HOUSE, "D", "CA", False),
    ("Tommy Tuberville", Chamber.SENATE, "R", "AL", True),
    ("Sheldon Whitehouse", Chamber.SENATE, "D", "RI", False),
    ("Ron Wyden", Chamber.SENATE, "D", "OR", False),
    ("Rick Scott", Chamber.SENATE, "R", "FL", False),
]


def seed_politicians(db: Session) -> int:
    created = 0
    for full_name, chamber, party, state, followed in SEED_POLITICIANS:
        exists = db.execute(
            select(Politician.id).where(Politician.full_name.ilike(full_name))
        ).first()
        if exists:
            continue
        db.add(
            Politician(
                full_name=full_name,
                chamber=chamber,
                party=party,
                state=state,
                followed=followed,
                active=True,
            )
        )
        created += 1
    db.commit()
    return created
