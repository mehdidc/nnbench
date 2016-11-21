import click
from train import train_model
import model_definitions
from lightjob.cli import load_db
from lightjob.db import AVAILABLE, PENDING, RUNNING, SUCCESS

import numpy as np
import os
import commentjson as json
import importlib

@click.group()
def main():
    pass


@click.command()
@click.option('--nb', default=1, required=False)
@click.option('--where', default='', required=False)
@click.option('--job_id', default='', required=False)
@click.option('--budget-hours', default=None, required=False)
def job(nb, where, job_id, budget_hours):
    db = load_db()
    kw = {}
    if where != '':
        kw['where'] = where
    if job_id != '':
        kw['summary'] = job_id
    print(kw)
    jobs = list(db.jobs_with(state=AVAILABLE, **kw))
    jobs = jobs[0:nb]
    for job in jobs:
        db.modify_state_of(job["summary"], PENDING)
    print('{} jobs pending...'.format(len(jobs)))
    for job in jobs:
        print('running {}'.format(job['summary']))
        db.modify_state_of(job["summary"], RUNNING)
        if budget_hours is not None:
            budget_hours = int(budget_hours)
            job['content']['optim']['budget_secs'] = budget_hours * 3600
        print(job['content']['optim'])
        train_and_save(db, job)
        db.modify_state_of(job["summary"], SUCCESS)

@click.command()
@click.option('--where', default='random', required=False)
@click.option('--nb', default=1, required=False)
def insert(where, nb):
    db = load_db()
    params_generator = params_sampler_from_python(where)
    for i in range(nb):
        params = params_generator()
        print(params)
        db.safe_add_job(params, where=where)

@click.command()
@click.option('--from-json', default=None, required=False)
@click.option('--from-python', default=None, required=False)
@click.option('--budget-hours', default=None, required=False)
@click.option('--outdir', default='out', required=False)
def run(from_json, from_python, budget_hours, outdir):
    np.random.seed(42)
    rng = np.random
    if from_json:
        params = json.load(open(from_json))
    elif from_python:
        sampler = param_sampler_from_python(from_python)
        params = sampler()
    train_model(params, outdir=outdir)

def param_sampler_from_python(label):
    tokens = label.split('.')
    module = '.'.join(tokens[0:-1])
    func = tokens[-1]
    module = importlib.import_module(module)
    return getattr(module, func)

def train_and_save(db, job, outdir=None):
    if outdir is None:
        outdir = os.path.join('out', 'jobs', job['summary'])
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    output = train_and_get_results(job['content'], outdir=outdir)
    db.update({'results': output}, job['summary'])

def train_and_get_results(params, outdir=None):
    model = train_model(params, outdir=outdir)
    output = model.history.history.copy()
    output.update(model.history.final)
    return output

if __name__ == '__main__':
    main.add_command(insert)
    main.add_command(run)
    main.add_command(job)
    main()
