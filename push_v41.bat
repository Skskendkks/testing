@echo off
REM skywager v4.1 — commit and push the code changes. No Python needed:
REM GitHub Actions does the data migration (next hourly poll) and retraining (nightly).
cd /d "%~dp0"
git add -A
git diff --cached --quiet && (echo nothing to commit) || git commit -m "v4.1: onset labels, 3-way split, consistent rain features, JTWC source fix, F3 tabular backfill (auto-migrates on CI)"
git push || goto :fail
echo.
echo Pushed. Next: GitHub -> Actions -> "poll" -> Run workflow   (migrates data now instead of waiting for the hour)
echo             GitHub -> Actions -> "retrain" -> Run workflow (retrains with the new method)
echo             GitHub -> Actions -> "backfill" -> Run workflow: mode=f3-events, start=2023-01-01, end=2026-08-01
start "" "https://github.com/Skskendkks/testing/actions"
pause
exit /b 0
:fail
echo.
echo *** git push failed - see above (are you signed in to GitHub? try GitHub Desktop).
pause
exit /b 1
