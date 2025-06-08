# embedding_generation_optimized.py - BEST PRACTICE VERSION

import os
import glob
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np

# Enhanced imports based on search results
from sentence_transformers import SentenceTransformer
import torch

# -----------------------------
# OPTIMIZED CONFIGURATION
# -----------------------------

# Model selection based on search results [3] and [5]
EMBEDDING_MODELS = {
    "production": {
        "name": "BAAI/bge-small-en-v1.5",  # Recommended in search results [3]
        "dimension": 384,
        "description": "Best balance of speed and quality"
    },
    "balanced": {
        "name": "all-mpnet-base-v2", 
        "dimension": 768,
        "description": "Good quality, moderate size"
    },
    "multilingual": {
        "name": "sentence-transformers/paraphrase-multilingual-mpnet-base-dot-v1",
        "dimension": 768,
        "description": "Multi-language support"
    }
}

# Configuration
SELECTED_MODEL = "production"  # Use BAAI/bge-small-en-v1.5 as recommended
MODEL_CONFIG = EMBEDDING_MODELS[SELECTED_MODEL]
MAX_CHARS_PER_CHUNK = 8000
OVERLAP_CHARS = 500
BATCH_SIZE = 32
USE_ONNX_OPTIMIZATION = True  # Based on search results [5]

class OptimizedEmbeddingGenerator:
    def __init__(self, model_config=MODEL_CONFIG, use_onnx=USE_ONNX_OPTIMIZATION):
        """Initialize optimized embedding generator"""
        self.model_config = model_config
        self.model_name = model_config["name"]
        self.expected_dim = model_config["dimension"]
        self.use_onnx = use_onnx
        self.model = None
        
        self._load_optimized_model()
    
    def _load_optimized_model(self):
        """Load model with ONNX optimization based on search results [5]"""
        print(f"🤖 Loading model: {self.model_name}")
        print(f"📏 Expected dimension: {self.expected_dim}")
        
        try:
            if self.use_onnx and torch.cuda.is_available():
                # ONNX optimization for GPU (from search results [5])
                print("⚡ Loading with ONNX GPU optimization...")
                self.model = SentenceTransformer(
                    self.model_name,
                    backend="onnx",
                    model_kwargs={"provider": "CUDAExecutionProvider"}
                )
            elif self.use_onnx:
                # ONNX optimization for CPU (from search results [5])
                print("⚡ Loading with ONNX CPU optimization...")
                self.model = SentenceTransformer(
                    self.model_name,
                    backend="onnx"
                )
            else:
                # Standard loading (from search results [3])
                print("📦 Loading standard model...")
                self.model = SentenceTransformer(self.model_name)
            
            # Verify dimension
            test_embedding = self.model.encode("test")
            actual_dim = len(test_embedding)
            
            if actual_dim != self.expected_dim:
                print(f"⚠️  Dimension mismatch: expected {self.expected_dim}, got {actual_dim}")
                self.expected_dim = actual_dim
            
            print(f"✅ Model loaded successfully - Dimension: {actual_dim}")
            
        except Exception as e:
            print(f"❌ ONNX loading failed: {e}")
            print("🔄 Falling back to standard model...")
            self.model = SentenceTransformer(self.model_name)
            test_embedding = self.model.encode("test")
            self.expected_dim = len(test_embedding)
            print(f"✅ Standard model loaded - Dimension: {self.expected_dim}")
    
    def chunk_text_smart(self, text, max_chars=MAX_CHARS_PER_CHUNK, overlap_chars=OVERLAP_CHARS):
        """
        Smart text chunking with sentence boundary detection
        """
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + max_chars, len(text))
            
            # Smart boundary detection
            if end < len(text):
                # Look for sentence endings
                for delimiter in ['. ', '.\n', '!\n', '?\n']:
                    last_delim = text.rfind(delimiter, start + max_chars//2, end)
                    if last_delim > start:
                        end = last_delim + len(delimiter)
                        break
                else:
                    # Look for paragraph breaks
                    last_para = text.rfind('\n\n', start + max_chars//2, end)
                    if last_para > start:
                        end = last_para + 2
                    else:
                        # Look for any newline
                        last_newline = text.rfind('\n', start + max_chars//2, end)
                        if last_newline > start:
                            end = last_newline + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            if end >= len(text):
                break
                
            start = end - overlap_chars
        
        return chunks
    
    def generate_embeddings_optimized(self, chunks):
        """
        Generate embeddings with optimization techniques from search results
        """
        try:
            if isinstance(chunks, str):
                chunks = [chunks]
            
            # Batch processing with progress bar (from search results [3])
            embeddings = self.model.encode(
                chunks,
                batch_size=BATCH_SIZE,
                show_progress_bar=len(chunks) > 5,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for better similarity search
            )
            
            return embeddings.tolist()
            
        except Exception as e:
            print(f"❌ Batch embedding failed: {e}")
            # Fallback to individual processing
            return [self.model.encode(chunk, normalize_embeddings=True).tolist() for chunk in chunks]
    
    def process_product_file(self, file_path, output_dir):
        """Process a single product file with enhanced metadata"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            return None
        
        # Smart chunking
        chunks = self.chunk_text_smart(text)
        product = os.path.splitext(os.path.basename(file_path))[0]
        
        print(f"📄 {product}: {len(chunks)} chunks ({len(text):,} chars)")
        
        # Generate embeddings
        start_time = time.time()
        embeddings = self.generate_embeddings_optimized(chunks)
        processing_time = time.time() - start_time
        
        print(f"⚡ {len(embeddings)} embeddings in {processing_time:.2f}s")
        
        # Load links
        links = self._load_links(file_path, product)
        
        # Enhanced metadata based on search results patterns
        result = {
            "product": product,
            "embeddings": embeddings,
            "chunks": chunks,  # Store original chunks for debugging
            "num_chunks": len(chunks),
            "links": links,
            
            # Model metadata
            "model_info": {
                "name": self.model_name,
                "dimension": self.expected_dim,
                "backend": "onnx" if self.use_onnx else "pytorch",
                "optimization": "enabled" if self.use_onnx else "disabled"
            },
            
            # Processing metadata
            "processing_info": {
                "chunk_method": "smart_boundary",
                "max_chars_per_chunk": MAX_CHARS_PER_CHUNK,
                "overlap_chars": OVERLAP_CHARS,
                "processing_time": processing_time,
                "embeddings_per_second": len(embeddings) / processing_time,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            
            # Quality metrics
            "quality_metrics": {
                "total_chars": len(text),
                "avg_chunk_size": np.mean([len(chunk) for chunk in chunks]),
                "chunk_size_std": np.std([len(chunk) for chunk in chunks])
            }
        }
        
        # Save with enhanced naming
        output_file = os.path.join(output_dir, f"{product}_embeddings_v2.json")
        try:
            with open(output_file, "w", encoding="utf-8") as out_f:
                json.dump(result, out_f, indent=2)
            print(f"✅ Saved: {output_file}")
            return product
        except Exception as e:
            print(f"❌ Save error for {product}: {e}")
            return None
    
    def _load_links(self, file_path, product):
        """Load links with fallback strategies"""
        # Try multiple link file patterns
        base_dir = os.path.dirname(file_path)
        link_files = [
            os.path.join(base_dir, f"{product}.links"),
            os.path.join(base_dir, f"{product}.json"),
            os.path.join(base_dir, f"{product}_links.txt")
        ]
        
        for links_file in link_files:
            if os.path.exists(links_file):
                try:
                    with open(links_file, "r", encoding="utf-8") as lf:
                        content = lf.read().strip()
                        try:
                            links = json.loads(content)
                            return links if isinstance(links, list) else [links]
                        except json.JSONDecodeError:
                            return [line.strip() for line in content.splitlines() if line.strip()]
                except Exception as e:
                    print(f"⚠️  Error reading {links_file}: {e}")
        
        return []

def main():
    """Main execution with enhanced reporting"""
    input_folder = "extracted_text"
    output_folder = "product_embeddings_v2"
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize generator
    generator = OptimizedEmbeddingGenerator()
    
    # Find files
    product_files = glob.glob(os.path.join(input_folder, "*.txt"))
    print(f"\n📁 Found {len(product_files)} product files")
    print(f"🤖 Model: {generator.model_name}")
    print(f"📏 Dimension: {generator.expected_dim}")
    print(f"⚡ Optimization: {'ONNX' if generator.use_onnx else 'Standard'}")
    
    # Process files with performance tracking
    processed_products = []
    failed_products = []
    start_time = time.time()
    total_embeddings = 0
    
    # Use ThreadPoolExecutor for parallel processing (from search results pattern)
    max_workers = min(4, len(product_files))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(generator.process_product_file, file_path, output_folder): file_path 
            for file_path in product_files
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                product = future.result()
                if product:
                    processed_products.append(product)
                    # Count embeddings for performance metrics
                    output_file = os.path.join(output_folder, f"{product}_embeddings_v2.json")
                    if os.path.exists(output_file):
                        with open(output_file, 'r') as f:
                            data = json.load(f)
                            total_embeddings += data.get('num_chunks', 0)
                else:
                    failed_products.append(futures[future])
            except Exception as e:
                failed_products.append(futures[future])
                print(f"❌ Failed: {futures[future]} - {e}")
    
    total_time = time.time() - start_time
    
    # Comprehensive summary
    print(f"\n🎉 **Processing Complete!**")
    print(f"✅ Successfully processed: {len(processed_products)} products")
    print(f"❌ Failed: {len(failed_products)} products")
    print(f"📊 Total embeddings generated: {total_embeddings:,}")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"⚡ Average speed: {total_embeddings/total_time:.1f} embeddings/second")
    print(f"💰 Estimated Azure cost saved: ${total_embeddings * 0.0001:.2f}")
    print(f"🏆 Model used: {generator.model_name} ({generator.expected_dim}D)")
    print(f"🚀 Optimization: {'ONNX enabled' if generator.use_onnx else 'Standard PyTorch'}")
    
    # Save processing report
    report = {
        "summary": {
            "processed_products": len(processed_products),
            "failed_products": len(failed_products),
            "total_embeddings": total_embeddings,
            "total_time": total_time,
            "embeddings_per_second": total_embeddings/total_time,
            "estimated_cost_saved": total_embeddings * 0.0001
        },
        "model_info": generator.model_config,
        "processing_config": {
            "max_chars_per_chunk": MAX_CHARS_PER_CHUNK,
            "overlap_chars": OVERLAP_CHARS,
            "batch_size": BATCH_SIZE,
            "onnx_optimization": generator.use_onnx
        },
        "processed_files": processed_products,
        "failed_files": failed_products
    }
    
    with open(os.path.join(output_folder, "processing_report.json"), "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()

