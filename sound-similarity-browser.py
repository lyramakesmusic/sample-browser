import os
import numpy as np
import torch
import laion_clap
from flask import Flask, request, render_template, jsonify, Response, send_file
from pathlib import Path
import librosa
import json
from typing import List, Dict, Tuple
import tempfile

class SoundSimilarityBrowser:
    def __init__(self, cache_file: str = "embeddings_cache.jsonl"):
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt()
        self.cache_file = cache_file
        self.embeddings_cache: Dict[str, List[float]] = {}
        self.load_cache()

    def load_cache(self):
        """Load cache from JSONL file"""
        if not os.path.exists(self.cache_file):
            return
            
        with open(self.cache_file, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    self.embeddings_cache[item['path']] = item['embedding']
                except Exception as e:
                    print(f"Error loading cache line: {e}")

    def _append_to_cache(self, path: str, embedding: List[float]):
        """Append a single embedding to the JSONL file"""
        with open(self.cache_file, 'a') as f:
            json.dump({'path': path, 'embedding': embedding}, f)
            f.write('\n')

    def process_folder(self, folder_path: str):
        """Generator that processes a folder and yields progress"""
        audio_files = [
            f for f in Path(folder_path).rglob("*")
            if f.suffix.lower() in ['.wav', '.mp3', '.ogg', '.flac']
        ]
        total_files = len(audio_files)
        
        for i, audio_file in enumerate(audio_files, 1):
            try:
                # Skip if already in cache
                if str(audio_file) in self.embeddings_cache:
                    yield i, total_files, str(audio_file), "skipped"
                    continue

                embedding = self.model.get_audio_embedding_from_filelist(
                    x=[str(audio_file)], 
                    use_tensor=True
                ).squeeze(0).detach().cpu().numpy().tolist()
                
                file_path = str(audio_file)
                self.embeddings_cache[file_path] = embedding
                self._append_to_cache(file_path, embedding)
                
                yield i, total_files, str(audio_file), "processed"
            
            except Exception as e:
                yield i, total_files, str(audio_file), f"error: {str(e)}"

    def find_similar(self, query_embedding: np.ndarray, top_n: int = 5) -> List[Tuple[str, float, List[str]]]:
        """Find top-N most similar sounds to the query embedding, grouping exact duplicates"""
        query_embedding = query_embedding.squeeze()
        
        # First group by identical embeddings
        embedding_groups = {}
        for path, cached_embedding in self.embeddings_cache.items():
            cached_embedding = np.array(cached_embedding)
            # Convert embedding to tuple so it can be used as dict key
            emb_key = tuple(cached_embedding)
            if emb_key not in embedding_groups:
                embedding_groups[emb_key] = {
                    'main_path': path,
                    'alt_paths': [],
                    'similarity': float(np.dot(query_embedding, cached_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
                    ))
                }
            else:
                embedding_groups[emb_key]['alt_paths'].append(path)

        # Convert to list format and sort by similarity
        unique_results = [
            (info['main_path'], info['similarity'], info['alt_paths'])
            for info in embedding_groups.values()
        ]
        
        return sorted(unique_results, key=lambda x: x[1], reverse=True)[:top_n]

    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text query"""
        embedding = self.model.get_text_embedding([text], use_tensor=True)
        return embedding.detach().cpu().numpy()

    def get_audio_embedding(self, audio_path: str) -> np.ndarray:
        """Get embedding for audio query"""
        embedding = self.model.get_audio_embedding_from_filelist(
            x=[audio_path], 
            use_tensor=True
        )
        return embedding.detach().cpu().numpy()

# Flask Application
app = Flask(__name__)
browser = SoundSimilarityBrowser()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process-folder', methods=['GET', 'POST'])
def process_folder():
    folder_path = request.args.get('folder_path') if request.method == 'GET' else request.form.get('folder_path')
    if not folder_path:
        return jsonify({'error': 'No folder path provided'}), 400
        
    def generate():
        for current, total, file_path, status in browser.process_folder(folder_path):
            data = {
                'current': current,
                'total': total,
                'file': file_path,
                'status': status,
                'progress': (current / total) * 100
            }
            yield f"data: {json.dumps(data)}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/serve-audio/<path:filepath>')
def serve_audio(filepath):
    return send_file(filepath)

@app.route('/search', methods=['POST'])
def search():
    if 'text' in request.form:
        # Text-based search
        query_text = request.form['text']
        query_embedding = browser.get_text_embedding(query_text)
    else:
        # Audio-based search
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file uploaded'}), 400
        
        audio_file = request.files['audio']
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            audio_file.save(temp_audio.name)
            query_embedding = browser.get_audio_embedding(temp_audio.name)
        os.unlink(temp_audio.name)

    similar_sounds = browser.find_similar(query_embedding)
    # Unpack the new format with alt_paths
    results = [(os.path.abspath(path), similarity, alt_paths) 
              for path, similarity, alt_paths in similar_sounds]
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)
