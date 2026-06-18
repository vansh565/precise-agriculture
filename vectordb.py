import chromadb
from chromadb.utils import embedding_functions
import numpy as np
from datetime import datetime
import json
import hashlib

class PlantDiseaseVectorDB:
    """Vector database for storing and matching plant disease conditions"""
    
    def __init__(self, db_path="./chroma_db"):
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Use sentence transformer for embeddings
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create or get collections
        self.disease_collection = self.client.get_or_create_collection(
            name="disease_records",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.similarity_collection = self.client.get_or_create_collection(
            name="disease_similarity",
            embedding_function=self.embedding_fn
        )
        
        print("✅ Vector Database initialized")
    
    def create_embedding_text(self, record):
        """Create a text representation for embedding"""
        return f"""
        Disease: {record.get('disease_name', '')}
        Severity: {record.get('severity', 0)}
        Temperature: {record.get('temperature', 0)}°C
        Humidity: {record.get('humidity', 0)}%
        Soil Moisture: {record.get('soil_moisture', 0)}%
        Spray Amount: {record.get('spray_amount', 0)} ml
        Treatment Outcome: {record.get('outcome', 'pending')}
        """
    
    def add_record(self, record):
        """Add a new disease detection record to vector DB"""
        try:
            # Create unique ID
            record_id = f"{record['timestamp']}_{hashlib.md5(record['disease_name'].encode()).hexdigest()[:8]}"
            
            # Create embedding text
            embedding_text = self.create_embedding_text(record)
            
            # Prepare metadata
            metadata = {
                "disease_name": record['disease_name'],
                "severity": record['severity'],
                "temperature": record['temperature'],
                "humidity": record['humidity'],
                "soil_moisture": record['soil_moisture'],
                "spray_amount": record['spray_amount'],
                "timestamp": record['timestamp'],
                "outcome": record.get('outcome', 'pending'),
                "confidence": record.get('confidence', 0)
            }
            
            # Add to collection
            self.disease_collection.add(
                ids=[record_id],
                documents=[embedding_text],
                metadatas=[metadata]
            )
            
            print(f"✅ Added to vector DB: {record['disease_name']} - {record_id}")
            return record_id
            
        except Exception as e:
            print(f"❌ Error adding to vector DB: {e}")
            return None
    
    def find_similar_conditions(self, query_record, n_results=5):
        """Find similar past conditions based on current query"""
        try:
            query_text = self.create_embedding_text(query_record)
            
            results = self.disease_collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            similar_records = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    similar_records.append({
                        "id": results['ids'][0][i],
                        "distance": results['distances'][0][i] if results['distances'] else 0,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "document": results['documents'][0][i] if results['documents'] else ""
                    })
            
            return similar_records
            
        except Exception as e:
            print(f"❌ Error finding similar conditions: {e}")
            return []
    
    def get_recommendation_from_similar(self, current_record):
        """Get spray recommendation based on similar past cases"""
        similar = self.find_similar_conditions(current_record, n_results=5)
        
        if not similar:
            return None
        
        # Calculate weighted average of spray amounts from similar cases
        total_weight = 0
        weighted_spray = 0
        total_outcome_score = 0
        
        for case in similar:
            # Weight by similarity (closer = higher weight)
            weight = 1 - case['distance']  # Convert distance to similarity
            total_weight += weight
            
            metadata = case['metadata']
            spray = metadata.get('spray_amount', 0)
            weighted_spray += spray * weight
            
            # Check outcome if available
            outcome = metadata.get('outcome', 'unknown')
            if outcome == 'successful':
                total_outcome_score += 2 * weight
            elif outcome == 'partial':
                total_outcome_score += weight
            elif outcome == 'failed':
                total_outcome_score -= weight
        
        avg_spray = weighted_spray / total_weight if total_weight > 0 else 0
        
        # Adjust based on similarity confidence
        confidence = min(1.0, total_weight / len(similar))
        
        return {
            "recommended_spray": round(avg_spray, 0),
            "confidence": round(confidence, 2),
            "similar_cases_found": len(similar),
            "outcome_score": round(total_outcome_score / len(similar), 2) if similar else 0,
            "similar_cases": similar
        }
    
    def update_outcome(self, record_id, outcome):
        """Update the outcome of a previous treatment"""
        try:
            self.disease_collection.update(
                ids=[record_id],
                metadatas=[{"outcome": outcome}]
            )
            print(f"✅ Updated outcome for {record_id}: {outcome}")
            return True
        except Exception as e:
            print(f"❌ Error updating outcome: {e}")
            return False
    
    def get_statistics(self):
        """Get vector database statistics"""
        count = self.disease_collection.count()
        return {
            "total_records": count,
            "collection_name": self.disease_collection.name,
            "embedding_model": "all-MiniLM-L6-v2"
        }
    
    def find_disease_trends(self, days=30):
        """Find disease trends over time"""
        # Get all records
        all_records = self.disease_collection.get()
        
        trends = {}
        for metadata in all_records['metadatas']:
            disease = metadata['disease_name']
            if disease not in trends:
                trends[disease] = {
                    "count": 0,
                    "avg_severity": 0,
                    "avg_spray": 0,
                    "success_rate": 0
                }
            trends[disease]["count"] += 1
            trends[disease]["avg_severity"] += metadata['severity']
            trends[disease]["avg_spray"] += metadata['spray_amount']
        
        # Calculate averages
        for disease in trends:
            count = trends[disease]["count"]
            trends[disease]["avg_severity"] /= count
            trends[disease]["avg_spray"] /= count
        
        return trends
    
    def clear_all_records(self):
        """Clear all records (for testing)"""
        self.disease_collection.delete(ids=self.disease_collection.get()['ids'])
        print("🗑️ All records cleared from vector DB")