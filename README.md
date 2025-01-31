# Chat3GPP: An Open-Source Retrieval-Augmented Generation Framework for 3GPP Documents

**Chat3GPP** is a Retrieval-Augmented Generation (RAG) framework for 3GPP documents, allowing for easy extension to other technical standards, providing a solid foundation for downstream tasks in telecommunications.

## References

- L. Huang, M. Zhao, L. Xiao, X. Zhang, J. Hu (2024). *Chat3GPP: An Open-Source Retrieval-Augmented Generation Framework for 3GPP Documents*. *arXiv preprint arXiv:2501.13954*. [Read the paper](https://arxiv.org/pdf/2501.13954.pdf)

## Getting Started

### Prerequisites

- Python 3.8
- Elasticsearch

### 1. Install Dependencies

Install the necessary Python packages:

```bash
pip install -r requirements.txt
```

### 2. Download & Preprocess Documents

Download the required 3GPP standard documents for this project. Once the 3GPP documents are downloaded, you need to preprocess them by removing unnecessary content. To do this, you can use the script `remove_content.py` located in the `preprocess` folder. After that, store the cleaned datas into the `knowledge_base` directory.

### 3. Store Data into Elasticsearch
After preprocessing the 3GPP documents, you need to import the cleaned data into an Elasticsearch database.

#### Steps to Import Data:

1. **Make sure Elasticsearch is running**: Ensure that your Elasticsearch instance is up and running. You need a working Elasticsearch server before proceeding.

2. **Configure Elasticsearch settings**: 
   
   - Open the `init_database.py` script.
   - Configure the connection settings for your Elasticsearch instance. Update the `host`, `port`, and authentication details, if necessary.

3. **Set up model paths**:
   
   - Open the `configs/model_configs.py` file.
   - Make sure that the paths to all required models are correctly specified. This file contains the paths to the machine learning models you plan to use.

4. **Store the data**: 
   
   - After configuring both Elasticsearch and the models, you can now run the script to import the preprocessed data into Elasticsearch.

   Run the following command from the root directory of the project:

   ```bash
   python init_database.py
   ```

### 5. Running the Model

Once the data is successfully stored into Elasticsearch, you can interact with the model in two ways:

- **Direct Conversation**: Use the `chat.py` script for a direct conversation with the model. 
- **Enhanced Model Generation with External Knowledge Base**: Use the `kb_chat.py` script to query an external knowledge base and enhance the model’s responses.
