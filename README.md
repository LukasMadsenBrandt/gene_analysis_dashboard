# Gene Analysis Dashboard

## Description

This project provides a dashboard for gene analysis using data from GENCODE and Kutsche datasets.

## Setup Instructions

### Prerequisites

- Python 3.6 or higher
- Graphviz installed and added to PATH

### Installation Instructions

#### Installing Python

**On Windows:**

1. Download Python from the official website: [Python Downloads](https://www.python.org/downloads/windows/).
2. Run the installer and ensure you check the box that says "Add Python to PATH".
3. Follow the installation steps.

**On macOS:**

1. Download Python from the official website: [Python Downloads](https://www.python.org/downloads/macos/).
2. Run the installer and follow the installation steps.
3. Alternatively, you can install Python using Homebrew:
    ```sh
    brew install python
    ```

**On Linux:**

1. Use the package manager for your distribution to install Python. For example, on Ubuntu:
    ```sh
    sudo apt update
    sudo apt install python3 python3-venv python3-pip
    ```

#### Installing Graphviz

**On Windows:**

1. Download Graphviz from the official website: [Graphviz Downloads](https://graphviz.org/download/).
2. Run the installer and follow the installation steps.
3. Add Graphviz to your system PATH (usually, the installer does this automatically).

**On macOS:**

1. Install Graphviz using Homebrew:
    ```sh
    brew install graphviz
    ```

**On Linux:**

1. Use the package manager for your distribution to install Graphviz. For example, on Ubuntu:
    ```sh
    sudo apt update
    sudo apt install graphviz
    ```

### Setting Up the Project

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```

2. **Create and Activate Virtual Environment**:

    **On Unix-like systems (Linux/macOS)**:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

    **On Windows**:
    ```cmd
    python -m venv venv
    venv\Scripts\activate
    ```

3. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Run the Application**:
    ```sh
    python app.py
    ```

## Usage

1. **Select the dataset and summarization technique**.
2. **Press "Send" to generate the graph**.
3. **Use the community detection options and other controls to customize the graph**.

## Contributing

Contributions are welcome. Please submit a pull request or open an issue to discuss the changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

