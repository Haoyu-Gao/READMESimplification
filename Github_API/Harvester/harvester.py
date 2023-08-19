from github import Github, RateLimitExceededException, GithubException
from credentials import *
import time

import argparse
import nltk

import csv
import json
import calendar
from loguru import logger
import pandas as pd
from pydriller import Repository

PAGE_SIZE = 500

simplification_set = set()

lemmatiser = nltk.WordNetLemmatizer()

with open("keywords.txt", 'r') as f:
    for line in f.readlines():
        simplification_set.add(line.strip())
print(simplification_set)


def github_api_data_harvester(db_name, id):
    ACCESS_TOKEN = ACCESS_TOKENS[id]
    start_id = 0 + 1000000 * id
    quit_id = start_id + 1000000
    g = Github(ACCESS_TOKEN, per_page=1000)
    # print(g.get_rate_limit())
    logger.info('started')
    cursor = 0

    while True:
        try:
            repos = g.get_repos(since=start_id,
                                visibility="all")  # this is all the available public repos sorted based on their created time
            count = 0
            try:
                for repo in repos:
                    count += 1
                    if count == PAGE_SIZE:
                        break
                    start_id = repo.id
                    if start_id > quit_id:
                        exit()
                    if repo.stargazers_count >= 10 and not repo.fork:
                        if repo.get_commits(path="README.md").totalCount != 0:
                            write_record(repo)

            except RateLimitExceededException as e:
                core_rate_limit = g.get_rate_limit().core
                reset_timestamp = calendar.timegm(core_rate_limit.reset.timetuple())
                sleep_time = reset_timestamp - calendar.timegm(time.gmtime()) + 15
                logger.info(f"exceeding rate limit, sleep for {sleep_time} seconds")
                logger.info(f"currently at {start_id}")
                time.sleep(sleep_time)
            except GithubException as e:
                count += 1
                continue
            except Exception as e:
                count += 1
                continue

        except RateLimitExceededException as e:
            core_rate_limit = g.get_rate_limit().core
            reset_timestamp = calendar.timegm(core_rate_limit.reset.timetuple())
            sleep_time = reset_timestamp - calendar.timegm(time.gmtime()) + 15
            logger.info(f"exceeding rate limit, sleep for {sleep_time} seconds")
            time.sleep(sleep_time)
        except GithubException as e:
            cursor += 1
            continue

        except Exception as e:
            cursor += 1
            continue


def keyword_filter():
    data = pd.read_csv("repo_before_keyword_filter.csv")
    data.columns = ["repo_name", "language", "star", "fork", "commit_count"]
    for i in range(len(data)):
        repo_url = f"https://github.com/{data.iloc[i]['repo_name']}.git"
        try:
            commit_message = None
            for commit in Repository(repo_url, filepath="README.md").traverse_commits():
                """
                see whether this commit only change the README.md file
                """
                src, dest, flag = None, None, None
                modified_files = commit.modified_files
                if len(modified_files) == 1:
                    message = commit.msg
                    if is_simplification_commit(message):
                        if src is None:  # the first one
                            src = modified_files[0].content_before
                            dest = modified_files[0].content
                            commit_message = message
                        else:
                            dest = modified_files[0].content
                            commit_message = message
        except:
            continue

        if src is not None:
            data_record = json.dumps({'repo_name': data.iloc[i]['repo_name'], 'lang': data.iloc[i]['language'],
                                      'stars': data.iloc[i]['star'], 'commit_count': data.iloc[i]['commit_count'],
                                      'src': src, 'dest': dest, 'commit_message': commit_message})

            with open("repo_after_keyword_filter.json", 'a') as f:
                f.write(data_record + "\n")


def write_record(repo):
    with open("repo_before_filter.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(
            [repo.full_name, repo.language, repo.stargazers_count, repo.fork, repo.get_commits().totalCount])


def is_simplification_commit(current_commit_message):
    """
    see if the current commit message is an indication of simplifying the texts.
    """
    tokenizer = nltk.SpaceTokenizer()
    tokens = tokenizer.tokenize(current_commit_message)

    for token in tokens:
        token = token.strip().rstrip('.').lower()
        lemma = lemmatiser.lemmatize(token, "v")
        if lemma == token:
            lemma = lemmatiser.lemmatize(token, "n")
            if lemma == token:
                lemma = lemmatiser.lemmatize(token, "a")

        if lemma.lower() in simplification_set:
            return True

    return False


# def gather_md_file_pairs(db_name, repo, commits, idx, last_md_url):
#     """
#     after verifying the current commit is a simplifying commit, we gather these two version of markdown files.
#     we decide to store them as json files and dumps into a CouchDB.
#     """
#     current_commit = commits[idx]
#     current_commit_url = current_commit.files[0].raw_url
#     current_commit_message = current_commit.commit.message
#     data_instance = None
#     client = DBClient(DB_USER, DB_PASSWORD, URL)

#     while not data_instance:
#         try:
#             data_instance = DataObject(last_md_url, current_commit_url, current_commit_message, repo, str(idx))

#             try:
#                 client.put_record(db_name, data_instance.to_json_format())

#             except Exception as e:
#                 print(e)
#         except:
#             continue


def remove_duplicate():
    repo_before = pd.read_csv("repo_before_filter.csv", header=None)
    repo_before.columns = ["repo_name", "language", "star", "fork", "commit_count"]
    repo_set = set()
    drop_idx = []
    for i in range(len(repo_before)):
        if repo_before.iloc[i]["repo_name"] in repo_set:
            drop_idx.append(i)
            continue
        repo_set.add(repo_before.iloc[i]["repo_name"])

    repo_before = repo_before.drop(drop_idx)
    repo_before.to_csv("repo_before_filter_drop_duplicate.csv", index=False, header=False)


if __name__ == "__main__":
    # print("?")
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, help="Specify the intended db name")
    parser.add_argument("--id", type=int, help="Specify the currently working process id")
    parser.add_argument("--size", type=int, help="Specify the size you want to get")
    args = parser.parse_args()
    db_name = args.db
    id = args.id
    size = args.size
    logger.add(f"harvester{id}.log")
    github_api_data_harvester(db_name, id, size)

