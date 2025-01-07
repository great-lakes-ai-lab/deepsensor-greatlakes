# DeepSensor-GreatLakes  

**DeepSensor-GreatLakes** is a specialized toolkit for applying machine learning methods, powered by [DeepSensor](https://deepsensor.readthedocs.io/), to Great Lakes research. This repository simplifies getting started with downscaling, sensor placement, interpolation, and filling missing data using publicly available datasets like GLSEA and ERA5.  

---

## Features  
- **Great Lakes Focus**: Preconfigured for GLSEA and other regional datasets.  
- **Multi-Platform Support**: Deploy on U-M HPC, Google Cloud Platform (GCP), or local machines.  
- **Reproducible Workflows**: Dockerized environment and configuration files for consistent execution.  
- **Use Cases**: Downscaling, sensor placement optimization, interpolation, and gap filling.  
- **Integrated Tools**: Includes support for [Weights & Biases](https://wandb.ai) for experiment tracking.  

---

## Getting Started  

### Prerequisites  
1. **Python 3.8+**  
2. [Docker](https://www.docker.com/get-started) (optional but recommended)  
3. Access to [GLSEA](https://www.glerl.noaa.gov/data/) and [ERA5](https://cds.climate.copernicus.eu/) datasets.  

### Installation  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/yourusername/deepsensor-greatlakes.git  
   cd deepsensor-greatlakes  
   ```
2. Set up a virtual environment an install dependencies:
   ```bash
   python3 -m venv venv  
   source venv/bin/activate  
   pip install -r requirements.txt  
   ```
## Repository Structure

```
deepsensor-greatlakes/  
├── README.md             # Project introduction and guide  
├── LICENSE               # Open-source license  
├── requirements.txt      # Python dependencies  
├── src/                  # Core Python code  
├── notebooks/            # Jupyter notebooks for demonstrations  
├── configs/              # Configuration files for reproducibility  
├── data/                 # Placeholder for datasets (gitignored)  
├── docker/               # Docker-related files  
└── tests/                # Unit tests for key components 
```

## Datasets  
- **GLSEA**: Great Lakes Surface Environmental Analysis (available in Zarr format).  
- **ERA5**: ECMWF reanalysis data for meteorological applications.  

Refer to `data/README.md` for instructions on accessing and preparing datasets.  

---

### Docker Setup for DeepSensor-GreatLakes

This guide walks you through the steps to build and run the Docker container for the DeepSensor-GreatLakes repository, which includes a Jupyter Notebook environment for hands-on work with Great Lakes data.

#### **1. Build the Docker Image**

To build the Docker image for DeepSensor-GreatLakes:

```bash
docker build -t my-deepsensor-app .
```

- The `-t` flag names the image (`my-deepsensor-app`).
- This process installs dependencies from the `requirements.txt` file and prepares the environment for running the app.

#### **2. Run the Docker Container**

To run the container and expose necessary ports (e.g., Jupyter Notebook):

```bash
docker run -it -p 8888:8888 my-deepsensor-app
```

- `-it` allows interactive terminal use.
- `-p 8888:8888` maps port 8888 in the container to port 8888 on your local machine, which is required for accessing Jupyter Notebook.

#### **3. Start Jupyter Notebook**

Once inside the container, start the Jupyter Notebook server:

```bash
jupyter notebook --ip=0.0.0.0 --allow-root
```

- `--ip=0.0.0.0` makes the server accessible from all IP addresses. 
- `--allow-root` grants the necessary permissions to run Jupyter as the root user in the container.

#### **4. Access Jupyter Notebook**

In your browser, open the following URL:

```
http://127.0.0.1:8888/tree?token=<your-token>
```

Replace `<your-token>` with the token provided in the terminal after running Jupyter. For convenience, you can disable the token requirement (recommended only for local, secure environments) by running Jupyter with:

```bash
jupyter notebook --ip=0.0.0.0 --allow-root --NotebookApp.token=''
```

#### **5. Additional Notes**
- The Docker image includes all necessary dependencies to work with Great Lakes data in DeepSensor.
- You can use the environment for model training, data exploration, and notebook-based workflows.
- Docker ensures consistency across different systems (e.g., local machines, cloud, HPC).

---

## Contributing  
Contributions are welcome! If you find a bug, have suggestions, or want to add new functionality:  
1. Fork the repository.  
2. Create a feature branch.  
3. Submit a pull request with a detailed description.  

---

## License  
This project is licensed under the [GNU Affero General Public License v3.0](LICENSE).  
 
---

## Acknowledgments  
This repository builds on the [DeepSensor](https://github.com/willirath/deepsensor) framework and is supported by resources from the University of Michigan and Google Cloud Platform.  

---

## Contact  
For questions or collaborations, feel free to open a discussion or reach out via email: your_email@example.com.  


