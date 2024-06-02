from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from chatbot import processTXT,processPDF,answer,getSerializedVectorStore
from waitress import serve
import os
import time
import uuid
import logging
from threading import Thread

DB_DIR = os.path.abspath('./vectorstores')
UPLOADS_DIR = os.path.abspath('./uploads')
EXPIRATION_TIME = 1800 # seconds

# Background cleanup process
def cleanupExpiredFiles():
    while True:
        current_time = time.time()
        for file_name in os.listdir(DB_DIR):
            file_path = os.path.join(DB_DIR, file_name)
            last_access_time = os.path.getatime(file_path)
            if current_time - last_access_time > EXPIRATION_TIME:
                os.remove(file_path)
        time.sleep(EXPIRATION_TIME)  # Check once every expiration period

# Start the background cleanup thread
cleanup_thread = Thread(target=cleanupExpiredFiles, daemon=True)
cleanup_thread.start()

logger = logging.getLogger('waitress')
logger.setLevel(logging.INFO)

app = Flask(__name__)

cors = CORS()
cors.init_app(app, resource={r"/api/*": {"origins": "*"}})

def saveSerializedVectorStore(serialized_vs):
    file_name = f"{uuid.uuid4()}"
    file_path = os.path.join(DB_DIR, file_name)
    with open(file_path, 'wb') as file:
        file.write(serialized_vs)
    return file_name

def loadSerializedVectorStore(file_name):
    file_path = os.path.join(DB_DIR, file_name)
    if not os.path.exists(file_path):
        return False
    
    with open(file_path, 'rb') as file:
        serialized_vs = file.read()
    os.utime(file_path, None)
    return serialized_vs

@app.route('/process', methods=['POST'])
def process():
    for fname in request.files:
        f = request.files.get(fname)
        
        if (f.content_type == 'application/pdf') | (f.content_type == 'text/plain'):
            secfname = secure_filename(fname)
            fpath = os.path.join(UPLOADS_DIR, secfname)
            f.save(fpath)

            chunks = []

            if f.content_type == 'application/pdf':
                chunks = processPDF(fpath)
            elif f.content_type == 'text/plain':
                chunks = processTXT(fpath)

            if os.path.exists(fpath):
                os.remove(fpath)

            if len(chunks) == 0:
                return 'Empty file', 400

            serialized_vs = getSerializedVectorStore(chunks)
            vs_id_fname = saveSerializedVectorStore(serialized_vs)

            return jsonify({"vector_store_id": vs_id_fname}), 200

        return 'Wrong file type', 415
    return 'No file supplied', 400

@app.route('/answer', methods=['POST'])
def askQuery():
    query = request.get_json()["query"]
    fname = request.get_json()["vector_store_id"]
    msgs = request.get_json()["msgs"]

    serialized_vs = loadSerializedVectorStore(fname)
    if serialized_vs == False:
        return 'Expired or non existent vector store', 404

    result = answer(serialized_vs, query, msgs)
    return jsonify(result), 200

if __name__ == "__main__":
    if not os.path.exists(UPLOADS_DIR):
        os.mkdir(UPLOADS_DIR)
    if not os.path.exists(DB_DIR):
        os.mkdir(DB_DIR)
    serve(app, host="0.0.0.0", port=5000, threads=2)