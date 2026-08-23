Authors: Cassie Cinzori, Amir Sesay, Maya Gayle, and Ian Solberg

# Applied_Econometrics_FinalProject

An applied econometrics final project (ECON 2560) asking how crude oil price
shocks affect household saving and borrowing patterns in the United States.
Gas prices are measured as the percent deviation from a 52 week moving
average, and that shock is regressed on consumer loan balances and bank
deposits over weekly data from June 2000 to September 2025, with unemployment
claims and the federal funds rate as controls.

This repository is also the group's shared code and filesharing space, which
is why the setup notes below are written for teammates rather than for
readers.

## Get these files on your computer

1. Install git if you don't have it. On macOS without Homebrew, run
   `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`,
   then `brew install git`.
2. Navigate to where you want this folder to live (short video tutorial:
   https://youtu.be/V4ShSik25Wo).
3. Click the green "Code" button at
   https://github.com/iasolb/Applied_Econometrics_FinalProject and copy the URL.
4. In Terminal run: `git clone <url>`.
5. Check in Finder that the folder is there.

## Get the latest updates

Once you have the repository on your computer, open Terminal, navigate to
the repository, and run `git pull`.

## Start here

Open `phase4/regression_analysis.py`. It builds the gas price shock variable
and runs the three OLS specifications behind the results, so it is the
shortest path from the raw series to the paper's tables. The inputs it reads
are in `final_data/`, and `data_transformation/data_cleaning.ipynb` shows how
those were assembled from the raw downloads.
