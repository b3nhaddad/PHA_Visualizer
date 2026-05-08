@echo off
REM Quick launcher — starts the frontend dev server.
REM Uses the portable Node.js downloaded during setup.
REM If Node is in your system PATH you can also just: cd frontend && npm run dev

set PATH=C:\Users\schmi\AppData\Local\node-portable;%PATH%
echo Starting Spherical Hessian Descent Visualizer (frontend only)...
echo Open http://localhost:5173 in your browser.
echo.
echo Tip: to run the full stack (frontend + backend) use Docker:
echo   docker compose up --build
echo.

cd frontend
npm install --silent
npm run dev
