# Simulation AI - Simulate the Future

Welcome to the forefront of scenario analysis. Gnerate insightful questions and responses, coupled with a dynamic interface for engaging with hypothetical scenarios. Ideal for strategists, researchers, and innovators, our tools facilitate the exploration of complex situations, enabling users to make well-informed decisions based on simuluted future scenarios. Embrace a realm of possibilities and endless number of future worlds with Simulation AI and revolutionize your approach to problem-solving and strategic planning.

This project was created for the [Assistants API LlamaIndex MongoDB Battle Hackathon](https://lablab.ai/event/assistants-api-llamaindex-mongodb-battle).

## Getting Started

### Prerequisites

Before running the scripts, ensure you have the following installed:
- Python 3.12
- MongoDB
- Streamlit
- OpenAI API key
- A Python environment manager (e.g., Miniconda, virtualenv)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/alexpwrd/aihackers.git
   ```
2. Navigate to the repository directory:
   ```bash
   cd aihackers
   ```
3. Set up a Python environment:
   You can use Miniconda or any other Python environment manager you prefer. Below are instructions for Miniconda and virtualenv as examples:

   - **Using Miniconda**:
     ```bash
     conda create --name sim python=3.12
     conda activate sim
     ```

   - **Using virtualenv**:
     ```bash
     python -m venv sim-env
     source sim-env/bin/activate  # On Windows use `sim-env\Scripts\activate`
     ```

4. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

Create a `.env` file in the root directory of the project with the following environment variables:
```
OPENAI_API_KEY=your_openai_api_key
MONGO_URI=your_mongodb_uri
```

#### Setting up MongoDB Atlas Search Index

To use the vector search capabilities, you need to set up an Atlas Search Index on your MongoDB collection. Follow these steps to create an index named `vector_index` for the `simulation.synthdata` collection:

1. Log in to your MongoDB Atlas dashboard.
2. Navigate to your cluster where the `simulation` database is hosted.
3. Go to the "Collections" tab, and select the `simulation` database and the `synthdata` collection.
4. Click on "Indexes" and then click "Create Index".
5. Choose "Search Index" and then use the following JSON configuration:
   ```json
   {
     "fields": [
       {
         "numDimensions": 1536,
         "path": "embedding",
         "similarity": "cosine",
         "type": "vector"
       }
     ]
   }
   ```
6. Name the index `vector_index` and create the index.

## Running the Scripts

### `sim_start.py` - Start Simulation

This script generates a series of questions based on a user-defined scenario and stores the responses in a MongoDB database.

#### How to Run

Execute the following command in your terminal:
```bash
python sim_start.py
```

Follow the prompts to enter the scenario you wish to explore. The script will handle the generation and storage of questions and their detailed responses.

### `sim_embed.py` - Embedding Creation

After generating data with `sim_start.py`, run this script to create embeddings for efficient querying.

#### How to Run

Execute the following command in your terminal:
```bash
python sim_embed.py
```

### `sim_chat.py` - Interactive Chat Interface

This script provides a web-based interface to chat with the stored responses.

#### How to Run

To start the Streamlit web interface, run:
```bash
streamlit run sim_chat.py
```

Navigate to the provided local URL in your web browser to interact with the application.

### `sim_eval.py` - Evaluation Interface

This script evaluates the responses using the TruLens framework.

#### How to Run

Execute the following command in your terminal:
```bash
python sim_eval.py
```

Navigate to the provided local URL in your web browser to interact with the application. Enter queries related to the stored scenarios and view responses directly from the MongoDB database.

## Usage Tips

- Ensure MongoDB is running and accessible via the URI provided in your `.env` file.
- Use the Streamlit interface for an interactive experience with real-time feedback.

## Support

For any issues or questions, please open an issue on the GitHub repository or contact us.

Thank you for using our Simulation AI and Query Interface tools!