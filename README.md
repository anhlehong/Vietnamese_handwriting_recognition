# Vietnamese Handwriting Recognition

A web application for recognizing handwritten Vietnamese text from images, integrated with Obsidian for digital note-taking. Built with Flask, TensorFlow, and Gemini API, this project uses a CRNN model with CTC loss to process handwritten text and generate Markdown files for seamless note management.

## Application Interface

1. **Home Page**  
   ![Home Page](https://res.cloudinary.com/dapvvdxw7/image/upload/v1750269266/home_ipz02k.png)

2. **Recognition Page**  
   ![Recognition Page](https://res.cloudinary.com/dapvvdxw7/image/upload/v1750269279/recognition_nyzbru.png)


## Prerequisites

- Python 3.10
- pip (Python package manager)
- A valid Gemini API key from Google Cloud Console

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/anhlehong/Vietnamese_handwriting_recognition.git
   cd Vietnamese_handwriting_recognition
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   - Create a file named `.env.local` in the **`app/routes`** folder.
   - Add the following content to `.env.local`, replacing `YOUR_GEMINI_API_KEY` with your own key:
     ```env
     GEMINI_API_URL=https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent
     API_KEY=YOUR_GEMINI_API_KEY
     DEBUG=True
     ```
   - To obtain a Gemini API key:
     1. Visit [Google Cloud Console](https://console.cloud.google.com/).
     2. Create a new project or select an existing one.
     3. Enable the **Generative Language API**.
     4. Generate an API key under **Credentials**.

4. **Prepare the CRNN Model**:
   - If the model is not downloaded, it will be automatically fetched when running the code.

## Usage

1. **Run the application**:
   ```bash
   python app.py
   ```
   - The web app will start in debug mode if `DEBUG=True` in `.env.local`.

2. **Access the web interface**:
   - Open your browser and navigate to `http://127.0.0.1:5000`.
   - Upload an image of handwritten Vietnamese text.
   - The app processes the image using the CRNN model, refines the output with Gemini API, and generates a Markdown file compatible with Obsidian.

3. **Integrate with Obsidian**:
   - Download the generated Markdown file.
   - Import it into your Obsidian vault to manage notes digitally.

## Project Structure

- **app/**:
  - **__pycache__/**: Compiled Python bytecode.
  - **model/**: Machine learning models.
  - **routes/**:
    - **__pycache__/**: Compiled Python bytecode.
    - **__init__.py**: Package initialization.
    - **.env.local**: Local environment variables.
    - **main.py**: Main script for routes.
  - **services/**: Service-layer logic.
  - **static/**: Static files (CSS, JS, images).
  - **templates/**: HTML templates.
  - **utils/**:
    - **__init__.py**: Package initialization.
  - **__init__.py**: Package initialization.
- **.gitignore**: Git ignore file.
- **app.py**: Main application script.
- **checkpoint_weights.weights.h5**: Model weights.
- **README.md**: Project documentation.
- **requirements.txt**: Dependencies list.
- **training.ipynb**: Jupyter Notebook for training.

## Dependencies

Key dependencies (listed in `requirements.txt`):
- Flask
- TensorFlow
- OpenCV
- Pillow
- NumPy
- Matplotlib
- Requests
- python-dotenv

## Security Notes

- **API Key Safety**: Never commit your `.env.local` file or API key to version control. Ensure `.env.local` is listed in `.gitignore`.
- **Gemini API**: Verify your API key has sufficient quota and is restricted to the Generative Language API for security.
- **Production**: Set `DEBUG=False` in `.env.local` for production to disable debug mode.

## Contributing

Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
