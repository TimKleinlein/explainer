from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse

import routes as routes

def start_app(db_uri='any'):
    app = FastAPI()

    # Register Blueprints
    app.include_router(routes.router)

    """
    @app.exception_handler(404)
    async def page_not_found(request, exc):
        return PlainTextResponse(str(exc.detail), status_code=exc.status_code)


    @app.exception_handler(500)
    async def internal_error(request, exc):
        return "Some Internal error has taken place."
    """

    app.mount("/static", StaticFiles(directory="static"), name="static")

    return app
