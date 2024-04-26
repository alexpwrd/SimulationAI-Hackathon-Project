# AI Hackers Project

This repository is dedicated to the AI Hackers project for Lablab.ai [Tool Face-Off: OpenAI Assistants API VS Llama-Index/MongoDB. An Eval-Driven Battle](https://lablab.ai/event/assistants-api-llamaindex-mongodb-battle).

## Prerequisites

Before you begin, ensure you have Miniconda installed on your machine. If not, you can download it from the [Miniconda Website](https://docs.conda.io/en/latest/miniconda.html) and follow the installation instructions for your operating system.

## Environment Setup

### Creating a Conda Environment

To create a Conda environment specifically for this project, open your terminal and execute the following command:

\```bash
conda create -n ai-hackers python=3.12
\```

After creating the environment, activate it by running:

\```bash
conda activate ai-hackers
\```

This will switch your terminal session to the `ai-hackers` environment.

## Git Workflow for Collaboration

### Creating and Managing Branches

Collaboration is key in software development. To manage changes and ensure the main branch remains stable, follow these steps:

1. **Create a new branch for each feature or bug fix:**
   Before starting on a new feature or fixing a bug, create a new branch to keep your changes organized and separate from the main branch.

   \```bash
   git checkout -b <branch-name>
   \```

   Replace `<branch-name>` with a descriptive name related to your task (e.g., `feature-login` or `bugfix-header`).

2. **Keep your branch updated:**
   Regularly pull changes from the main branch to keep your branch up-to-date, reducing conflicts during merges.

   \```bash
   git pull origin main
   \```

3. **Commit and push your changes:**
   Make regular commits with descriptive messages and push your changes to the remote repository frequently.

   \```bash
   git add .
   git commit -m "Describe your changes here"
   git push origin <branch-name>
   \```

4. **Create a pull request:**
   Once your feature or fix is complete, initiate a pull request (PR) to merge your branch into the main branch, allowing team members to review the changes.

   - Navigate to your repository on GitHub.
   - Click on 'Pull Requests'.
   - Click 'New Pull Request'.
   - Select your branch and the main branch to compare.
   - Enter details about the PR and submit it.

5. **Review and merge the pull request:**
   Have at least one other team member review the PR. Address any comments or required changes. Once approved, merge the PR.

   - Click 'Merge Pull Request' once it's approved.
   - Delete the branch if it's no longer needed.

### Best Practices

- **Communicate with your team**: Keep everyone updated on what you are working on and any issues you encounter.
- **Test thoroughly**: Ensure your changes are well tested before merging to avoid introducing bugs.
- **Review code**: Engage in code reviews to maintain high code quality and learn from each other.

## Contribution

Feel free to contribute to this project by following the Git workflow outlined above and let's try to win this hackathon.
