# cython_optimizations.pyx
import torch  # Make sure torch is imported
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def process_image_batch(object batch_data, object model, object processor):
    """
    Processes a batch of page images.
    
    Parameters:
      - batch_data: list of tuples (pdf_name, page_idx, page_img)
      - model: a loaded ColIdefics3 model.
      - processor: a loaded ColIdefics3Processor.
      
    Returns:
      A list of dicts, each containing:
        - 'id': Unique ID (format: pdf_name_page_{page_idx})
        - 'embedding': Embedding vector as list of floats
        - 'metadata': A dict with keys 'pdf_name' and 'page_idx'
        - 'document': An empty string.
    """
    cdef list results = []
    cdef int n = len(batch_data)
    cdef list images = []
    cdef list meta_list = []
    cdef int i

    # Extract images and corresponding metadata.
    for i in range(n):
        images.append(batch_data[i][2])
        meta_list.append((batch_data[i][0], batch_data[i][1]))
    
    # Process images as a batch (calls are Python-level).
    cdef object inputs = processor.process_images(images).to("cpu")
    
    # Declare emb_output outside the with block.
    cdef object emb_output = None
    with torch.no_grad():
        emb_output = model(**inputs)
    
    # Mean pool over the sequence dimension (assumed dim=1).
    cdef object pooled = emb_output.mean(dim=1)
    
    for i in range(n):
        pdf_name, page_idx = meta_list[i]
        results.append({
            "id": "%s_page_%d" % (pdf_name, page_idx),
            "embedding": pooled[i].tolist(),
            "metadata": {"pdf_name": pdf_name, "page_idx": page_idx},
            "document": ""
        })
    
    return results

