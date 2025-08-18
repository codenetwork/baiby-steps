
# 🤖 b{AI}by Steps – Train Your Own 3D Walker

Welcome to **b{AI}by Steps**, a reinforcement learning project where club members create, train, and test custom 3D walkers using Python, Gymnasium, and physics simulators. The walkers are trained to navigate challenging terrains and evaluated based on performance, robustness, and creativity.

---

## 🎯 Project Goals

- Build multiple 3D walkers (humanoids, bipeds, quadrupeds, etc.)
- Create and modify diverse terrain environments
- Train policies using reinforcement learning (PPO, A2C)
- Evaluate and compare walker performance across terrains
- Showcase walker competition at semester’s end 🎉

---

## 🧠 Technologies

- [Python 3.10+](https://www.python.org/)
- [Gymnasium](https://gymnasium.farama.org/)
- [PyBullet](https://pybullet.org/wordpress/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- NumPy, Matplotlib

---
##  🙋 How to Contribute
We welcome contributors of all skill levels!
- Pick an issue from GitHub Issues
- Write README, issues, and instructions
- Add new walkers (`walkers/`)
- Create terrain environments (`terrains/`)
- Improve the training/evaluation scripts
- Visualize reward curves or gait animations

---
## 📖 Recommended Learning
We'll be using the gymnasium library to create our environments. I recommend reading the gymnasium documentation and following the steps to create some of the simpler pre-existing environments - 
[here](https://gymnasium.farama.org/introduction/basic_usage/)

You can find some info on creating URDF robots [here](https://articulatedrobotics.xyz/tutorials/ready-for-ros/urdf/#overall-structure---links-and-joints)

Here are some good YouTube videos too:
- [The FASTEST introduction to Reinforcement Learning on the internet
](https://www.youtube.com/watch?v=VnpRp7ZglfA&t=4155s)

## 🔧 How to Get Started

### 1. Clone the Repository

```bash
git clone https://github.com/codenetwork/baiby-steps
cd baiby-steps
```
### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
