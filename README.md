A repository to collect utility code for the WiDS Datathon 2025: Unraveling the Mysteries of the Female Brain: Sex Patterns in ADHD.

# Guide to files
- [connectome.py](connectome.py): defines WiDSDataset, a torch-geometric dataset of the functional connectome graphs provided by WiDS.

# Installation
## Clone the git repository
This will put a copy of the git repository on your computer.
```sh
git clone https://github.com/pmfirestone/widsdatathon2025
```
## Setting up Kaggle Authentication
To use the WiDSDataset, you have to authenticate with kaggle. [Here](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md#api-credentials) are the instructions on how to do that.

## Installing prerequisites
From the base directory of the repository, run:
```sh
pip install -r requirements.txt
```
I recommend you use a virtual environment for this.
# Resources
## Articles
- [Loh et al., Automated detection of ADHD: Current trends and future perspective](https://www.sciencedirect.com/science/article/abs/pii/S0010482522003171)
- [Zhao et al., A dynamic graph convolutional neural network framework reveals new insights into connectome dysfunctions in ADHD](https://www.sciencedirect.com/science/article/pii/S1053811921010466)
- [Five articles from WiDS Community](https://drive.google.com/drive/folders/14flM-1i7Ksz3iKY2KWhnlBz8RNguJBCA)
## Libraries
- https://pytorch-geometric.readthedocs.io/
## Resources from WiDS Community
- https://community.widsworldwide.org/posts/global-datathon-hub-2025-challenge-resources-and-tutorials

# License
See the [LICENSE](LICENSE.md) file for license rights and limitations (MIT).
