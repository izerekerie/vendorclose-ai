"""
Database Module for VendorClose AI
Handles storage of uploaded training data
"""

import sqlite3
import os
from datetime import datetime
from pathlib import Path
import json


class TrainingDataDB:
    """SQLite database for storing uploaded training data"""
    
    def __init__(self, db_path='data/training_data.db'):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table for uploaded images
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS uploaded_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                class_label TEXT NOT NULL,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                used_for_training BOOLEAN DEFAULT 0,
                training_session_id TEXT
            )
        ''')
        
        # Table for training sessions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                status TEXT DEFAULT 'pending',
                metrics TEXT,
                model_path TEXT
            )
        ''')
        
        # Table for model metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_image(self, filename, filepath, class_label):
        """
        Add uploaded image to database
        
        Args:
            filename: Original filename
            filepath: Path where image is stored
            class_label: Class label (fresh/medium/rotten)
            
        Returns:
            Image ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO uploaded_images (filename, filepath, class_label)
            VALUES (?, ?, ?)
        ''', (filename, filepath, class_label))
        
        image_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return image_id
    
    def get_unused_images(self):
        """Get all images not yet used for training"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, filename, filepath, class_label, uploaded_at
            FROM uploaded_images
            WHERE used_for_training = 0
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': r[0],
                'filename': r[1],
                'filepath': r[2],
                'class_label': r[3],
                'uploaded_at': r[4]
            }
            for r in results
        ]
    
    def mark_images_as_used(self, image_ids, session_id):
        """
        Mark images as used for training
        
        Args:
            image_ids: List of image IDs
            session_id: Training session ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        placeholders = ','.join(['?'] * len(image_ids))
        cursor.execute(f'''
            UPDATE uploaded_images
            SET used_for_training = 1, training_session_id = ?
            WHERE id IN ({placeholders})
        ''', [session_id] + image_ids)
        
        conn.commit()
        conn.close()
    
    def create_training_session(self, session_id):
        """
        Create a new training session
        
        Args:
            session_id: Unique session identifier
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO training_sessions (session_id, status)
            VALUES (?, 'in_progress')
        ''', (session_id,))
        
        conn.commit()
        conn.close()
    
    def update_training_session(self, session_id, status, metrics=None, model_path=None):
        """
        Update training session
        
        Args:
            session_id: Session ID
            status: New status
            metrics: Dictionary of metrics
            model_path: Path to saved model
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        completed_at = datetime.now().isoformat() if status == 'completed' else None
        metrics_json = json.dumps(metrics) if metrics else None
        
        cursor.execute('''
            UPDATE training_sessions
            SET status = ?, completed_at = ?, metrics = ?, model_path = ?
            WHERE session_id = ?
        ''', (status, completed_at, metrics_json, model_path, session_id))
        
        conn.commit()
        conn.close()
    
    def get_training_sessions(self, limit=10):
        """Get recent training sessions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT session_id, started_at, completed_at, status, metrics, model_path
            FROM training_sessions
            ORDER BY started_at DESC
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'session_id': r[0],
                'started_at': r[1],
                'completed_at': r[2],
                'status': r[3],
                'metrics': json.loads(r[4]) if r[4] else None,
                'model_path': r[5]
            }
            for r in results
        ]
    
    def get_statistics(self):
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total images
        cursor.execute('SELECT COUNT(*) FROM uploaded_images')
        total_images = cursor.fetchone()[0]
        
        # Images by class
        cursor.execute('''
            SELECT class_label, COUNT(*) 
            FROM uploaded_images 
            GROUP BY class_label
        ''')
        images_by_class = dict(cursor.fetchall())
        
        # Used images
        cursor.execute('SELECT COUNT(*) FROM uploaded_images WHERE used_for_training = 1')
        used_images = cursor.fetchone()[0]
        
        # Training sessions
        cursor.execute('SELECT COUNT(*) FROM training_sessions')
        total_sessions = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_images': total_images,
            'images_by_class': images_by_class,
            'used_images': used_images,
            'unused_images': total_images - used_images,
            'total_sessions': total_sessions
        }

