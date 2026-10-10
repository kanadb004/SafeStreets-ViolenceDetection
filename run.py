"""Development server for the SafeStreets demo app (docs/SPRINT_PLAN.md Sprint 5).

    python run.py

Serves `safestreets.web.create_app()` on port 3000, as the legacy `app` package did.
"""

from safestreets.web import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=False, port=3000)
