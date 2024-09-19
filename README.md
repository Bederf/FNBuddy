<<<<<<< HEAD
Here's the updated version of your README file that includes the information on the OpenAI API key requirement, following the structure you've provided:

---

# FNBuddy

**FNBuddy** is an AI-powered office assistant designed to help with answering facility-related queries. In its current version, FNBuddy retrieves information from local files stored in the `/info` folder and provides real-time assistance to users. Future versions may include additional office solutions, such as integration with SharePoint for document retrieval and broader office management functionalities.

## Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Planned Features](#planned-features)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## About the Project

FNBuddy is designed to assist with answering common facility-related queries using AI. The current version references information from local files stored in the `/info` folder. While future versions may include features like SharePoint integration for document retrieval, FNBuddy currently focuses on being a reliable AI assistant for office-related queries.

## Features

- AI-powered responses using OpenRouter’s language models.
- Answers questions based on information stored in the `/info` folder.
- Easily expandable to accommodate future office-related functionalities.

## Technologies Used

- **Python**: Backend programming.
- **Flask**: Web framework for creating the API.
- **OpenRouter API**: For generating AI-powered responses.
- **FAISS**: Vector store for efficient document searches.
- **HTML/CSS**: Frontend interface.
- **JavaScript**: For frontend interactions.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or above
- pip (Python package manager)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Bederf/FNBuddy.git
   cd FNBuddy
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the necessary dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the `.env` file for environment variables:**

   The `.env` file should already contain the `OPENROUTER_API_KEY` required for the application. However, you must obtain your own OpenAI API key:

   - **OpenAI API Key:** You need to sign up on [OpenAI](https://platform.openai.com/) and get your API key.

   Add the following line to your `.env` file, replacing the placeholder with your actual OpenAI API key:

   ```plaintext
   OPENROUTER_API_KEY="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"
   OPENAI_API_KEY=your_openai_api_key
   ```

   Ensure to replace `your_openai_api_key` with your actual OpenAI API key.

5. **Run the Flask app:**

   ```bash
   python fnbuddy-info.py
   ```

   Your local app should now be running on `http://127.0.0.1:5006/`.

## Usage

1. **Answering Queries**: Users can ask questions, and FNBuddy will respond based on the information in the `/info` folder.

## Planned Features

- **SharePoint Integration**: Future versions may include the ability to retrieve and reference documents stored in SharePoint.
- **Total Office Solution**: FNBuddy may evolve into a full office assistant with more comprehensive features such as document management, scheduling, and facility operations.

## Contributing

If you'd like to contribute to FNBuddy, follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix:

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Make your changes and commit them:

   ```bash
   git commit -m "Add your commit message here"
   ```

4. Push your changes to GitHub:

   ```bash
   git push origin feature/your-feature-name
   ```

5. Open a pull request and describe your changes.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

For any questions or feedback, feel free to reach out:

- **Pieter Van Rooyen** - [GitHub](https://github.com/Bederf) - [Email](mailto:bederf@gmail.com)

---

Let me know if there are any further adjustments you'd like to make!
=======
# FNBuddy
>>>>>>> 859708572c62127eca6ae8bb5ccd7b2113efc401
