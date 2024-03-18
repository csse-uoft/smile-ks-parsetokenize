# smile_ks_parsetokenize
SMILE Knowledge Source: ParseTokenize



# Setup

#### 1 GraphDB Installation
- [GraphDB](https://www.ontotext.com/products/graphdb/) can be installed for your distribution.
- Make sure it's running port `7200`, e.g. [http://localhost:7200](http://localhost:7200).
- Make sure you have GraphDB running on [http://localhost:7200](http://localhost:7200).

#### 2 GraphDB Repository Creation
- For testing, make sure the username and password are set to `admin`
- Create a new test repository. Go to [http://localhost:7200](http://localhost:7200)
  - Create new repository:
    - Name the repository (Repository ID) as `smile`
    - Set context index to True *(checked).
    - Set query timeout to 45 second.
    - Set the `Throw exception on query timeout` checkmark to True (checked)
    - Click on create repository.
  - Make sure the repository rin in "Running" state.
- [See instruction below for troubleshooting](#user-content-graphdb-and-docker-configuration)


#### 3 GraphDB Configuration
A few notes on configurting SMILE to connect to the database.
- The main SMILE Flask application can be configured in the [config/local_config.yml](config/local_config.yml) file.
- Knowledge source that are running in a Docker instance must use the "Docker" version of the config file: [config/local_config_test.yml](config/local_config_test.yml).



#### 4 Install Prolog
Install [swi-prolog](https://www.swi-prolog.org/download/stable) for your environment.


### 5. Install StanfordCoreNLP packages
  - `cd src/smile_ks_parsetokenize/libs/corenlp/`
  - `chmod +x nlp_server.sh`
  - `curl https://downloads.cs.stanford.edu/nlp/software/stanford-corenlp-4.3.1.zip > stanford-corenlp-4.3.1.zip`
  - `open stanford-corenlp-4.3.1.zip`
  - `curl https://downloads.cs.stanford.edu/nlp/software/stanford-parser-4.2.0.zip > stanford-parser-4.2.0.zip`
  - `open stanford-parser-4.2.0.zip`
  - `cd ../../../..`

#### 4 Setup smile-ks-qa1
`cd src/smile_ks_parsetokenize`
`conda env create -f PyParseTokenize.yml`
`./scripts/setup_folders.sh`


## To run example
You will need two terminals.

Term 1, run:
`cd src/smile_ks_parsetokenize/libs/`
`./corenlp/nlp_server.sh`
This will start a NLP server at [http://localhost:9000](http://localhost:9000). It will run in the background.

Term 2: Run ParseTokenize example
`conda activate PyParseTokenize`
`cd src`
`python -m smile_ks_parsetokenize.main`

## To run KnowledgeSource Listener
`conda activate PyParseTokenize`
`cd src`
`python -m smile_ks_parsetokenize.listener`
