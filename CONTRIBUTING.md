# Contributing

Contributions to the project can be made in the form of issues or pull requests. We welcome and encourage you to contribute to this project. Some areas of contribution include but are not limited to:

- **Code**
  - Development of new features
  - Refactoring of existing code
  - Bug fixes
- **Documentation**
  - Adding to or improving the existing documentation
- **Tests**
  - Writing new regression tests
  - Extending existing tests
  - Testing the code on different platforms and reporting any issues
- **Support**
  - Answering questions or getting involved in disscussion on the [Issues](https://github.com/NSAPH-Projects/space/issues) or [Discussions](https://github.com/NSAPH-Projects/space/discussions)


## Getting Started

You do not need to be part of the core team to contribute to this project. If you are new to open source, you can look for issues tagged as `good first issue` or `help wanted` to get started. If you are looking for a place to start, you can also look at the [open issues](https://github.com/NSAPH-Projects/space/issues) to see if there is anything that interests you.

A simple contribution steps are as follows:

- Fork the repository
  - The easiest way to get started is to fork the repository to your own GitHub account. This will create a copy of the repository under your account that you can use to make changes.
  - Go to the repository on GitHub and click the "Fork" button in the top right corner. This will create a copy of the repository under your account.
  

- Clone the repository to your local machine
  - After forking the repository, you will need to clone the repository to your local machine.
  - To clone the repository, open a terminal and run the following command, replacing `<username>` with your GitHub username:
  
    ```sh
    git clone git@github.com:<username>/space.git
    ```

- Set up the environment
  - After cloning the repository, you will need to set up the environment to run the code.
  - To set up the environment, you numerous options:
    - Option 1: Using `pip` to install the dependencies.
      - First, create a new virtual environment using `venv`:
        ```sh
        python -m venv env
        ```
      - Then, activate the virtual environment:
          ```sh
          source env/bin/activate
          ```
      - Finally, install the dependencies using `pip`:
          ```sh
          pip install -r requirements.txt
          ```
    
    - Option 2: Using `conda` to create a new environment and install the dependencies.
      - First, create a new environment using `conda`:
        ```sh
        conda create -n space python=3.10
        ```
      - Then, activate the environment:
        ```sh
        conda activate space
        ```
      - Finally, install the dependencies using `conda`:
        ```sh
        conda install --file requirements.txt
        ```
    - Option 3 (recommended):Using `docker` to run the code in a container.
        - TBD

- Get the most updated code from the main repository
  - Before making any changes, you should make sure that you have the most updated code from the main repository.
  - To get the most updated code, you can add the main repository as a remote and pull the changes from the main repository:
    ```sh
    git remote add space git@github.com:NSAPH-Projects/space.git
    ```
  - Then fetch the changes from the space repository:
    ```sh
    git fetch space
    ```
  - Finally, checkout to your own `dev` branch, merge the changes from the space repository's `dev` branch into your local `dev` branch:
    ```sh
    git merge space/dev
    ```
- Run the tests
  - Before making any changes, you should run the tests to make sure that the code is working as expected.
  - To run the tests, you can use the following command:
    ```sh
    pytest .
    ```
  - It should pass all the tests without any errors or failures, otherwise, there might be an issue with your environment. 

- Make your changes
  - After setting up the environment and running the tests, you can make your changes.
  - We recommend that you create a new branch for your changes to keep your changes separate from the main codebase.
  - Please make sure that your changes are consistent with the coding style and conventions used in the project.
  - Please dedicate each commit to a single change and provide a meaningful commit message.
  - Run the tests after making your changes to make sure that your changes did not break anything.
  - Update `CHANGELOG.md` with a brief description of your changes.

- Push your changes to your fork
  - After making your changes, you will need to push your changes to your fork.
  - To push your changes, you can use the following command:
    ```sh
    git push origin <branch-name>
    ```

- Open a pull request
  - After pushing your changes to your fork, you can open a pull request to the `space` repository.
  - To open a pull request, go to the main repository on GitHub and click the "New pull request" button.
  - Make sure that you are comparing the changes from your fork to the `dev` branch of the main repository.
  - Provide a meaningful title and description for your pull request.
  - If you need to get feedback on your changes and also leverage the GitHub Actions, you can add the `WIP` (work in progress) label to your pull request. This will let others know that your pull request is not ready to be merged yet.
  - After opening the pull request, you can wait for the pull request to be reviewed and merged.
  - In case you need to make changes to your pull request, you can make the changes in your local branch and push the changes to your fork. The pull request will be updated automatically with the new changes.
  - Once your pull request is ready to be merged, you can remove the `WIP` label and request a review from the maintainers.
  - After your pull request is reviewed and approved, it will be merged into the main repository.
  - Please note that your pull request will be automatically tested using GitHub Actions. If the tests fail, you will need to fix the issues before your pull request can be merged. So, please keep an eye on the tests. If your tests fail, the PR will be ignored and will not be reviewed until the tests pass. Unless you have a good reason for the tests to fail, in that case, put a comment on the PR to let the maintainers know.
  - Failing to follow the guidelines, making changes that are not consistent with the coding style, or not providing tests for your changes may result in your pull request being rejected.

- Done!


## Notes

- **Git Branching Model**
  - We follow the Successful Git Branching Model convention for our branching model. We have two main branches: `main` and `dev`. The `main` branch is the main branch that contains the stable code. The `dev` branch is the development branch that contains the latest changes. You can read more about the Successful Git Branching Model [here](https://nvie.com/posts/a-successful-git-branching-model/).
  - It is the responsibility of the contributor to make sure that their changes are consistent with the `dev` branch. If your changes are not consistent with the `dev` branch, you will need to rebase your changes to the `dev` branch before opening a pull request.
- **Coding Style**
  - We follow the PEP 8 coding style convention for our code. Please make sure that your changes are consistent with the PEP 8 coding style convention.
