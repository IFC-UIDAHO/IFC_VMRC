VMRC Seedling Mortality Web Application Overview
_______________________________________________________________________________________________________________________________
_______________________________________________________________________________________________________________________________

The VMRC Seedling Mortality Web Application is an internal research tool for visualizing and analyzing modeled seedling mortality under different vegetation cover levels, species types, climate scenarios, and seasonal conditions.

Users can:
================

Draw or upload an Area of Interest (AOI)

Select species, vegetation cover %, climate scenario, and month

Generate clipped mortality rasters

View raster overlays on an interactive map

Retrieve summary statistics for the selected AOI

The system consists of a static frontend and a FastAPI backend service.

Application Architecture
=================================================

Frontend

HTML / CSS / JavaScript

Leaflet for interactive mapping

Static files served via web server (e.g., Nginx)

Communicates with backend via HTTP API

Backend

Python 3.x

FastAPI (ASGI framework)

Uvicorn (ASGI server)

rasterio

numpy

shapely


Application instance:

app.main:app

Repository Structure
backend/
app/
  main.py
frontend/
requirements.txt
README.md

Raster datasets (~85GB) are NOT included in this repository.

Raster Data Requirements

Total size: ~85GB

Raster data must be stored separately from the repository.

Expected structure:

Mortality/
    Douglas_Fir/
    Western_Hemlock/
    HighStressMortality/
=============================================================================

Required Environment Variable
RASTER_ROOT

Absolute path to the root mortality raster directory.

Example (Linux server):

export RASTER_ROOT=/data/Mortality

Example (Windows local development):

set RASTER_ROOT=D:\Mortality

The backend reads this variable at startup.

==========================================================================

Local Development Setup

1. Clone Repository
git clone <repo_url>
cd <repo_folder>
___________________________________________________________


2. Create Virtual Environment
python -m venv .venv

Activate:

Windows:

.venv\Scripts\activate

Mac/Linux:

source .venv/bin/activate
___________________________________________________________


3. Install Dependencies
pip install -r requirements.txt

___________________________________________________________

4. Set Raster Root 

Windows:

set RASTER_ROOT=D:\Mortality

Mac/Linux:

export RASTER_ROOT=/path/to/Mortality
___________________________________________________________

5. Run Backend 
python -m uvicorn app.main:app --reload --port 8000

Backend will be available at:

http://localhost:8000
___________________________________________________________

6. Run Frontend

If static:

cd frontend
python -m http.server 3000

Open:

http://localhost:3000


Install Dependencies
pip install -r requirements.txt

Configure Environment Variable
export RASTER_ROOT=/data/Mortality

Start Backend (Production Mode)

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
_____________________________________________________________________

Updating the Application

Code is managed via IFC GitHub.

Hosting pulls latest changes.


Important Notes

Raster datasets must not be committed to GitHub.

Ensure RASTER_ROOT is configured before starting backend.


Support

For deployment coordination or configuration clarification, please contact the development team.