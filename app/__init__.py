import os
from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'footages'

    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    return app
