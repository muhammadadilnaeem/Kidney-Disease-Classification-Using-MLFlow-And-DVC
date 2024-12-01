

------

# **`Step by Step Implementation of Kidney Disease Project`**

1. Create a **GitHub** repository.
    - Add **.gitignore**, **licence** and **README.md**

2. Create a **template.py** file which will help us to create a Project template with single command **(Instead of making files and folders manually)**. here is how this file will look like:

    ```bash
    # import libraries
    import os
    import logging
    from pathlib import Path

    # set up logging string
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

    # set project name
    project_name = "kidney_disease_classifier"

    # specify files and folders to be created
    list_of_files = [
        ".github/workflows/.gitkeep",
        f"src/{project_name}/__init__.py",
        f"src/{project_name}/components/__init__.py",
        f"src/{project_name}/utils/__init__.py",
        f"src/{project_name}/config/__init__.py",
        f"src/{project_name}/config/configuration.py",
        f"src/{project_name}/pipeline/__init__.py",
        f"src/{project_name}/entity/__init__.py",
        f"src/{project_name}/constants/__init__.py",
        "config/config.yaml",
        "dvc.yaml",
        "params.yaml",
        "Dockerfile"
        "requirements.txt",
        "setup.py",
        "streamlit.py",
        "research/trials.ipynb",
        "templates/index.html",

    ]

    # Iterate over each file path in the list of files
    for filepath in list_of_files:

        # Convert the filepath string to a Path object for easier manipulation
        filepath = Path(filepath)

        # Split the filepath into its directory and filename components
        filedir, filename = os.path.split(filepath)

        # Check if the directory part of the filepath is not empty
        if filedir !="":

            # Create the directory if it doesn't exist, without raising an error if it does
            os.makedirs(filedir, exist_ok=True)

            # log the meaasge that the file and directory created
            logging.info(f"Creating directory; {filedir} for the file: {filename}")

        # Check if the file does not exist or is empty
        if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
            
            # Open the file in write mode (this creates the file if it doesn't exist)
            with open(filepath, "w") as f:

                # No content is written to the file
                pass

                # log the message that created an empty firl    
                logging.info(f"Creating empty file: {filepath}")

        else:

            # Log that the file already exists
            logging.info(f"{filename} is already exists")
    ```
    - To implement this structure go to terminal and type:
    
    ```bash
    python tepmlate.py 
    ```

    - You will see automatically you folder structure will be created.

4. We need to add required libraries for this project in **requirements.txt**. for this prject we used the following:

    ```bash
    tensorflow
    pandas 
    dvc
    mlflow
    notebook
    ipykernel
    numpy
    matplotlib
    seaborn
    python-box
    pyYAML
    tqdm
    ensure
    joblib
    types-PyYAML
    scipy
    Flask
    Flask-Cors
    streamlit
    gdown
    -e .
    ```

3. Now if we want to set up our folder as local package we need to write this code in **setup.py** which will look like this:

    ```bash
    # Import the setuptools library to facilitate packaging and distribution
    import setuptools

    # Open the README file to read its content for the long description, using UTF-8 encoding
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

    # Define the version of the package
    __version__ = "0.0.0"

    # Define metadata for the package
    REPO_NAME = "Kidney-Disease-Classification-Using-MLFlow-And-DVC"  # GitHub repository name
    AUTHOR_USER_NAME = "muhammadadilnaeem"  # GitHub username
    SRC_REPO = "kidney_disease_classifier"  # Name of the source directory/package
    AUTHOR_EMAIL = "madilnaeem0@gmail.com"   # Contact email for the author

    # Call the setup function to configure the package
    setuptools.setup(
        name=SRC_REPO,  # Name of the package, must match the directory in 'src'
        version=__version__,  # Version of the package
        author=AUTHOR_USER_NAME,   # Author name as it will appear in package metadata
        author_email=AUTHOR_EMAIL,  # Author's contact email
        description="A demo python package for Kidney Disease Classification Web Application.",  # Short package description
        long_description=long_description,  # Detailed description read from README.md
        long_description_content="text/markdown",  # Format of the long description (markdown)
        url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",  # URL of the package repository
        project_urls={   # Additional URLs related to the project, such as an issue tracker
            "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
        },
        package_dir={"": "src"}, # Specify that the root of packages is the 'src' directory
        packages=setuptools.find_packages(where="src")  # Automatically find all packages in 'src'
    )
    ```

4. Now if i want to set up **src** folder as my local package i need to add `-e .` at the end of my **requirements.txt** file. You do not need to run **setup.py** seperately.   

5. Now we need to create a **Virtual Environment**. This will help us to avoid libraries conflict. 

    - For creating virtual environment use this command:

        ```bash
        conda create -p venv python=3.10 -y
        ```

    - Now we need to **activate** the created Virtual Environment. For this use this command:

        ```bash
        conda activate /workspaces/Kidney-Disease-Classification-Using-MLFlow-And-DVC/venv
        ```
        - In case if you see any error 
            
            - Close the current terminal or shell window and open a new one. This will allow the changes from conda init to take effect.

            - run this command again.

6. Next step would be to set up **logging and exception** for better code readibilty and pracrice.

    - We need to set up **logging**.

        - Inside `src` folder we have `__init__.py`. We will write logging code in this file.It will help us direcly import `logger`. Here is what logging code will look like:

            ```bash
            import os  # Import the os module for interacting with the operating system
            import sys  # Import the sys module for system-specific parameters and functions
            import logging  # Import the logging module for logging messages

            # Define the logging format string
            logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

            # Specify the directory where log files will be stored
            log_dir = "logs"

            # Create the full path for the log file
            log_filepath = os.path.join(log_dir, "running_logs.log")

            # Create the log directory if it does not exist
            os.makedirs(log_dir, exist_ok=True)

            # Configure the logging settings
            logging.basicConfig(
                level=logging.INFO,  # Set the logging level to INFO
                format=logging_str,  # Use the defined format for log messages
                handlers=[
                    logging.FileHandler(log_filepath),  # Log messages to a file
                    logging.StreamHandler(sys.stdout)    # Also output log messages to the console
                ]
            )

            # Create a logger object with a specific name
            logger = logging.getLogger("kidney_disease_classifier_logger")  # This logger can be used throughout the application
            ```

        - We need to **set up exceptions**. For this project We will set this up using a python package `python-box`. All exceptions code can be found in `common_functions.py` which we will discuss in next step. But first we will use some new packages for this prject and we need to take a look:

        ### **python-box**

        - **Explanation**:  
        `python-box` is a versatile library that extends Python dictionaries, allowing you to access dictionary keys as object attributes (dot-notation). It also includes advanced functionality, such as working seamlessly with JSON, YAML, and other data structures.

        - **Use Case**:  
        - Ideal for managing configuration data or JSON-like structures in an easy-to-access and user-friendly manner.

        - **Example**:  
        ```python
        from box import Box

        # Original dictionary
        dic = {"name": "John", "age": 25}

        # Convert to a Box object
        dic = Box(dic)

        # Access values using dot notation
        print(dic.name)  # Output: John
        print(dic.age)   # Output: 25

        # Adding new key-value pairs
        dic.country = "USA"
        print(dic.country)  # Output: USA
        ```

        - **Additional Features**:
        - Handles nested dictionaries for easy access.
        - Converts JSON/YAML files directly into Box objects for seamless integration with configuration files.


        ### **config-box**
        - **Explanation**: 
        - `config-box` provides a way to handle Python dictionaries as objects, enabling dot-notation access (e.g., `obj.key`) instead of traditional dictionary indexing (`obj["key"]`). It is especially helpful when working with configuration files like YAML or JSON.

        - **Use Case**: 
        - Simplifies working with nested dictionary configurations.

        - **Example**:
        ```python
        from box import ConfigBox

        # Original dictionary
        dic = {"key": "value", "tala": "chabi"}

        # Convert to ConfigBox
        dic = ConfigBox(dic)

        # Access value with dot notation
        print(dic.tala)  # Output: chabi
        ```

        ### **ensure**
        - **Explanation**: 
        - `ensure` enforces type annotations at runtime. If the provided arguments don't match the annotated types, it raises an error. This is particularly useful for maintaining strict type-checking in functions.

        - **Use Case**: 
        - Ensures functions only receive arguments of expected types.

        - **Example**:
        ```python
        from ensure import ensure_annotations

        @ensure_annotations
        def get_multiplication(x: int, y: int):
            return x * y

        # Correct usage
        print(get_multiplication(x=6, y=9))  # Output: 54

        # Incorrect usage
        print(get_multiplication(x=6, y="9"))  # Raises TypeError: argument 'y' must be int
        ``` 

6. Now it's time to create **functions** that w**e will use again and again in our project**. Instead of writing code again and again for different functions we will just write code and keep it in `common_functions.py`, whenever we need any functionality we will just import that code from this file. 
   - For this I will go to `src` and then `utils` folder.
   - Inside this folder we will create a file `common_functions.py`, which will look like this
        ```bash
        # Import libraries
        import os
        import json
        import yaml
        import joblib
        import base64
        from typing import Any
        from pathlib import Path
        from box import ConfigBox
        from ensure import ensure_annotations
        from box.exceptions import BoxValueError
        from chest_cancer_classifier import logger

        @ensure_annotations
        def read_yaml(path_to_yaml: Path) -> ConfigBox:
            """
            Reads a YAML file and returns its contents as a ConfigBox object.

            Args:
                path_to_yaml (Path): Path to the YAML file.

            Raises:
                ValueError: If the YAML file is empty.
                Exception: If any other error occurs during file reading.

            Returns:
                ConfigBox: Parsed content from YAML as a ConfigBox object.
            """
            try:
                with open(path_to_yaml) as yaml_file:
                    content = yaml.safe_load(yaml_file)
                    logger.info(f"YAML file '{path_to_yaml}' loaded successfully")
                    return ConfigBox(content)
            except BoxValueError:
                logger.error(f"YAML file '{path_to_yaml}' is empty")
                raise ValueError("YAML file is empty")
            except Exception as e:
                logger.exception(f"Error loading YAML file '{path_to_yaml}': {e}")
                raise e


        @ensure_annotations
        def create_directories(path_to_directories: list, verbose=True):
            """
            Creates a list of directories if they do not already exist.

            Args:
                path_to_directories (list): List of directory paths to create.
                verbose (bool, optional): If True, logs creation for each directory. Defaults to True.
            """
            for path in path_to_directories:
                os.makedirs(path, exist_ok=True)
                if verbose:
                    logger.info(f"Directory created at: {path}")


        @ensure_annotations
        def save_json(path: Path, data: dict):
            """
            Saves a dictionary as a JSON file.

            Args:
                path (Path): Path where JSON file will be saved.
                data (dict): Data to save in JSON format.
            """
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
            logger.info(f"JSON file saved at: {path}")


        @ensure_annotations
        def load_json(path: Path) -> ConfigBox:
            """
            Loads JSON data from a file and returns it as a ConfigBox object.

            Args:
                path (Path): Path to the JSON file.

            Returns:
                ConfigBox: Data loaded from JSON file as a ConfigBox.
            """
            with open(path) as f:
                content = json.load(f)
            logger.info(f"JSON file loaded successfully from: {path}")
            return ConfigBox(content)


        @ensure_annotations
        def save_bin(data: Any, path: Path):
            """
            Saves data in binary format using joblib.

            Args:
                data (Any): Data to save as binary.
                path (Path): Path where binary file will be saved.
            """
            joblib.dump(value=data, filename=path)
            logger.info(f"Binary file saved at: {path}")


        @ensure_annotations
        def load_bin(path: Path) -> Any:
            """
            Loads binary data using joblib.

            Args:
                path (Path): Path to the binary file.

            Returns:
                Any: Object stored in the binary file.
            """
            data = joblib.load(path)
            logger.info(f"Binary file loaded from: {path}")
            return data


        @ensure_annotations
        def get_size(path: Path) -> str:
            """
            Gets the file size in kilobytes (KB).

            Args:
                path (Path): Path of the file.

            Returns:
                str: Size in KB, rounded to the nearest integer.
            """
            size_in_kb = round(os.path.getsize(path) / 1024)
            logger.info(f"Size of '{path}': ~{size_in_kb} KB")
            return f"~ {size_in_kb} KB"


        def decodeImage(imgstring: str, fileName: str):
            """
            Decodes a base64 string and saves it as an image file.

            Args:
                imgstring (str): Base64-encoded image string.
                fileName (str): Path where the decoded image will be saved.
            """
            imgdata = base64.b64decode(imgstring)
            with open(fileName, 'wb') as f:
                f.write(imgdata)
            logger.info(f"Image decoded and saved to '{fileName}'")


        def encodeImageIntoBase64(croppedImagePath: str) -> str:
            """
            Encodes an image file as a base64 string.

            Args:
                croppedImagePath (str): Path to the image file.

            Returns:
                str: Base64-encoded string of the image file.
            """
            with open(croppedImagePath, "rb") as f:
                encoded_string = base64.b64encode(f.read())
            logger.info(f"Image at '{croppedImagePath}' encoded to base64")
            return encoded_string.decode('utf-8')
        ```

7. Now we need to understand **data Acquisition** for this. You can get data from any where like from databases (mongodb,sql), platforms like (kaggle, uci machine learning).
    - But for this project we will acquire data from `Google Drive`.

8. Now we have **set up project**, We need to **understand how to write our code step by step**.  

    ### **Project Workflow**

    0. **Perform Experiments in `research` folder**:
   
    - For every stage we will create seperate **Jupyter notebook**, we will write code in **Jupyter notebook (.ipynb)** for **experiments** and use same code in **python files (.py)**. Here are the names of these files:
   
      - `01_data_ingestion.ipynb`         
      - `02_prepare_base_model.ipynb`         
      - `03_model_trainer.ipynb`         
      - `04_model_evaluation.ipynb`         

    1. **Update `config.yaml`**:  
    Define project-wide static configurations, such as file paths, thresholds, or model parameters.

    2. **Update `secrets.yaml` (Optional)**:  
    Store sensitive data like API keys or passwords securely (if required).

    3. **Update `params.yaml`**:  
    Define tunable parameters for experiments, such as learning rates or batch sizes.
       - We will load parameters of `VGG16` 

    4. **Update the Entity**:  
    Create structured classes to represent data models, ensuring type safety and clarity.
        - In our case we have file `config_entity.py`. This file has all entities used in this project for configuration.

    5. **Update the Configuration Manager**:  
    Implement a utility in `src/config` to parse and manage configuration files dynamically.
        - In our case we have `configuration.py`.

    6. **Update the Components**:  
    Write modular functions or classes in `src/components` to perform specific tasks like data preprocessing or model training. Here are the folloowing components used in this project:
        - `data_ingestion.py`.
        - `prepare_base_models.py`.
        - `model_trainer.py`.
        - `model_evaluation_with_mlflow.py`.

    7. **Update the Pipeline**:  
    Integrate components in `src/pipeline` to define end-to-end workflows (e.g., data processing to model inference).
     - Our pipeline stages step by step will be 
      
        - `stage_1_data_ingestion.py`
        - `stage_2_prepare_base_model.py`
        - `stage_3_model_training.py`
        - `stage_4_model_evaluation.py`
        - `stage_5_prediction.py`

    8.  **Update the `main.py`**:  
    Write the entry point script to execute the pipeline and orchestrate the workflow.

    9.  **Update the `dvc.yaml`**:  
    Define and version your data pipeline with DVC, including stages for data preparation, model training, and evaluation.

### **Experimentation and Pipeline Tracking**

9. We can **track experiments** that we perform with our code using **Open Source** `MLops Tools` like **Dagshub**, **MLflow** and **DVC**.

    - For this project we will first set up our Dagshub account and then use our **github repository** in **Dagshub** for **experiment tracking using MLflow**.

        - We also need to store our secret **dagshub** credentials. For this we will use **`.env`** file. Our credentials will look like this

            ```bash
            MLFLOW_TRACKING_URI="https://dagshub.com/muhammadadilnaeem/Kidney-Disease-Classification-Using-MLFlow-And-DVC.mlflow"
            MLFLOW_TRACKING_USERNAME="muhammadadilnaeem"
            MLFLOW_TRACKING_PASSWORD="d77b3df886771c7712d09e78e20a"
            ```

        - Then for tracking we will write code. example of MLflow code is:
      
            ```bash
            def log_into_mlflow(self):
                    # Set the MLflow tracking URI from the configuration
                    mlflow.set_registry_uri(self.config.mlflow_uri)
                    
                    # Parse the tracking URI to determine the type of storage used
                    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
                    
                    # Set the experiment name
                    experiment_name = "Model Evaluation Experimentation" 
                    mlflow.set_experiment(experiment_name)

                    # Start a new MLflow run to log parameters and metrics
                    with mlflow.start_run(run_name="Model Evaluation"):
                        # Log parameters from the configuration
                        mlflow.log_params(self.config.all_params)
                        # Log evaluation metrics (loss and accuracy)
                        mlflow.log_metrics(
                            {"loss": self.score[0], "accuracy": self.score[1]}
                        )
                        # Check if the tracking URL is not a file store
                        if tracking_url_type_store != "file":
                            # Register the model with MLflow
                            mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
                                            
                        else:
                            # Log the model without registration if using file store
                            mlflow.keras.log_model(self.model, "model")
            ```

          - You need to remember that **mlflow** is just for **tracking of best parameters** of model. Once you get best parameters you need to **comment off that part of code**. Like:

              ```bash
              evaluation.log_into_mlflow()
              ``` 

        - Now to integrate **DVC** for **Pipeline Tracking** () use these commands:

            - First step is to initialize DVC:
                
                ```bash
                dvc init
                ```

            - Second step will be to run this command for pipeline tracking which will run dvc.yaml to perform this:
             
                ```bash
                dvc repro
                ```

            - Now to see graph of how the whole pipeline connected (dependecy of the pipeline) run this command:

                ```bash
                dvc dag
                ```            

### **AWS-CICD-Deployment-Wiith-Github-Actions**

10. Now we Need to Containerize our Application. For this we will use `Docker`. In order to do this we first need to create a `Dockerfile` then we will write code in it:
 
    ```bash
    # Use the official Python 3.8 slim image as the base image
    FROM python:3.8-slim-buster

    # Update the package list and install the AWS CLI
    RUN apt update -y && apt install awscli -y

    # Set the working directory inside the container to /app
    WORKDIR /app

    # Copy the current directory contents into the /app directory in the container
    COPY . /app

    # Install the Python dependencies listed in requirements.txt
    RUN pip install -r requirements.txt

    # Specify the command to run the application
    CMD ["python3", "streamlit.py"]
    ```
    -  After building docker Image of source code we will push that `Dockerfile` in **AWS ECR** (Amazon Elastic Container Registry).

    -  We will launch **EC2**.

    -  Then we will pull Dockerimage from **ECR in **EC2**. 

    -  After this we will launch Dockerimage in **EC2**.


11. **Create an IAM (Identity and Access Management)** :
 
    - Log into **AWS Console**. Search **IAM** in search bar and Select it.

    - Click on **Users**, and Create new user by pressing **Create User** button.
     
      - Set **User name**.
        - In our case Let's name it **kidney disease classifier** and click on **Next** button.

    - Next **Set Permissions** tab will appear.

      - In **Permissions options** select
        - **Attach policies directly** and then search and select these 2 policies and then Press **Next** button.
         
          - **AmazonEC2ContainerRegistryFullAccess**             
          - **AmazonEC2FullAccess**
      
      - In **Review and create** tab just press **Create user** button. Your user will be created.
      
      - Now you will go to **users** click **kidney disease classifier** user.
        
        - Then go to **Security credientials** and scroll down and Select **Access keys** and press **Create access key** button.
        
          - Then select **Command Line Interface (CLI)** and press **Next** button. Then on next page press **Create access keys** button.
          
          - It is better to **Download .csv file** which has your credentials. and then Press **Done** button.

13. **Create an ECR (Elastic Container Registry) Repository** :
 
    - Search **ECR** in search bar and Select it.
      
      - Click on **Get Started** button. 
      - **General Settings** page will appear.
        
        - Select **Private** and give a **repository name** in our case it will be **kidney disease classifier**, then hit **Create repository** button.
      
      - Now Copy the **URI** and save it somewhere safe we will use it later.
      - Also keep note of your **region**.

13. **Create EC2 instance (Virtual server in the AWS Cloud)**
    
    - Search **EC2** in search bar and Select it.
      
      - Press **Launch Instance** button.
      1. Now name machine in our case it will be **kidney disease classifier**.
      2. Select **Ubuntu** machine.
      3. Select instance type in our case let's say we have **t2.medium**.
      4. Next Select **Create new key pair**. name the same **kidney disease classifier** and hit **Create key pair** button.
      5. In **Network settings** select checkbox:
        - **Allow HTTPS traffic from the internet**                                                      
        - **Allow HTTP traffic from the internet**
      6. In **Storage** tab you can set size. In our case let's keep it **10 GB**.
      7. Hit **Launch instance** button.
      8. You will see **Instances** tab. Wait for few seconds. When it shows machine status running.
      9. Click on **Instance ID** and it will take you on new page.
      10. Press **Connect** button.
      11. Now we need to run few commands which are:

        ```bash
        # Optional commands

        # update the machine
        sudo apt-get update -y

        sudo apt-get upgrade

        # Required commands

        # install docker in the ubuntu machine
        curl -fsSL https://get.docker.com -o get-docker.sh

        # installs Docker with superuser privileges
        sudo sh get-docker.sh

        # add user "ubuntu" to the "docker" group
        sudo usermod -aG docker ubuntu

        # activate new group permissions for "docker."
        newgrp docker

        # verify docker is working fine
        docker --version
        ```   

    - Now we need to **Configure EC2 as self-hosted runner** for this go to **github project repository settings**:
      
      - Click on **Actions** Option, select **Runners**.
        - Click on **New self hosted runner**.
        - In next page select **linux** and execute the below commands one by one in running **EC2** machine:

        ```bash
        # Download

        # Create a folder
        $ mkdir actions-runner && cd actions-runner

        # Download the latest runner package
        $ curl -o actions-runner-linux-x64-2.321.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.321.0/actions-runner-linux-x64-2.321.0.tar.gz

        # Optional: Validate the hash
        $ echo "ba46ba7ce3a4d7236b16fbe09423fb453bc08f866b24f04d549ec89f1722a29e  actions-runner-linux-x64-2.321.0.tar.gz" | shasum -a 256 -c

        # Extract the installer
        $ tar xzf ./actions-runner-linux-x64-2.321.0.tar.gz

        ---------------------------------------------------

        # Configure

        # Create the runner and start the configuration experience
        $ ./config.sh --url https://github.com/muhammadadilnaeem/Chest-Cancer-Classification-Using-MLflow-and-DVC --token BGZ6LQQ7T4NQM6V7KTRFELFSFJC62S

        # 1. for runner group just press enter button.

        # 2. for runner name enter "self-hosted"

        # 3. again press enter.
        
        # 4. again press enter.


        # Last step, run it!
        $ ./run.sh

        ---------------------------------------------------

        # optional

        # Using your self-hosted runner

        # Use this YAML in your workflow file for each job
        runs-on: self-hosted
        ```

        - To verify this go back to github in **Runners** You will see **self-hosted** runner status as **Idle**.

14. **Setting Up Github Secrets** : 

    - Go back to your project repository settings.
      
      - Go to **Secrets and variables** tab.
        - Click on **Action**
          - Click on **New Repository secret**
          - Add following Secrets

          ```bash
          AWS_ACCESS_KEY_ID=

          AWS_SECRET_ACCESS_KEY=

          AWS_REGION= us-east-1

          AWS_ECR_LOGIN_URI= 

          ECR_REPOSITORY_NAME= kidney disease classifier
          ```    
15. If you want to add a specific **port number**,
    
    - Go back to your **EC2** instance.
        
        - Select **Security** tab.
        - In same tab click on **Security groups** which will look like numbers.
            - Click on **Edit inbound rules** button.
            - Click on **Add rule** button.
            - Infront of **Cuntom TCP** enter your port range suppose we have **8080** and **0.0.0.0** and click **save rules** button.
        
        - Go back to your **EC2** Select **Running Machine** and copy **Public iPv4 address**.

16. Go to google and past address and add **port number** at the end
    
    ```bash
    http://650.122.251:8080
    ```    

    - Demonstarate your web app works **fine**.

17. Then at the end to avoid the extra charger terminate the **instance**. for this
    - Select the machine, click on **Instance State** and select **terminate instance** and press **Terminate** button.

18. Also delete your **ECR**.
19. Also delete your **IAM**.





----