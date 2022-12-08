# IMPORTS

from flask import Flask, request, session, render_template

import torch
from torch.optim import Adam

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import VariationalELBO

# from matplotlib import pyplot as plt
import numpy as np
import secrets
import gc
# import time
import os
import errno
import sys
import csv

sys.path.insert(0, os.getcwd() + '/modules')  
from customDataset import CustomDataset as customDataset
from sound import Stimulus as stimulus
from acquisition import BALD as BALD
from acquisition import Random as Random
from util import move_s
from util import RMSELoss
from twoAFC import TwoAFC as twoafc
from psychometric_curve import PsychometricCurve

# INITIALIZE FLASK APP
secret = secrets.token_urlsafe(32)
app = Flask(__name__)
app.secret_key = secret
torch.set_flush_denormal(True)
#gc.set_debug(gc.DEBUG_LEAK)
gc.enable()

# plt.switch_backend('Agg')

# CLASSES AND METHODS

# REMOVE FILES
def silentremove(filename):
    try:
        print('deleting ' + filename)
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred

# TODO DEFINE PSYCHOMETRIC LATENT FUNCTION

ALPHA = 25 # threshold
BETA = 10 # slope
# GAMMA = 0.5 # guess rate
DELTA = 0.01 #lapse rate

def PF_test_function(x):
    y = 1 - torch.exp( - (x / ALPHA) ** BETA)
    pf = 0.5 * (1 - DELTA) * y + 0.5
    return pf

# INITIALIZE MODEL AND LIKELIHOOD
# TODO CHECK VARIATIONAL STRATEGY FOR APPROXIMATION
# TODO CHECK KERNEL

class GPClassificationModel(ApproximateGP):
    def __init__(self, train_x):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(self, train_x, 
            variational_distribution, learn_inducing_locations=True)
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred

# TRAIN AND TEST METHODS

def train(model, likelihood, optimizer, training_iterations, train_data, mll):
    # startTrain = time.time()
    model.train()
    likelihood.train()
    trainX = train_data.inputs
    trainY = train_data.labels
    for i in range(training_iterations):
        optimizer.zero_grad(set_to_none=True)
        output = model(trainX)
        loss = -mll(output, trainY)
        loss.backward()
        # print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()
    # endTrain = time.time()
    #print('TRAIN TIME', endTrain - startTrain)

def test(model, likelihood, test_data, criterion):
    # startTest = time.time()
    model.eval()
    likelihood.eval()
    # test_loss = 0
    # correct = 0
    # lenghtTest = test_data.inputs.numel()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_data.inputs))
        #pred_labels = observed_pred.mean.ge(0.5).float()
        #score = criterion(pred_labels, test_data.labels)
        # print('RMSE SCORE: ', score)
    # endTest = time.time()
    # print('TEST TIME', endTest - startTest)
    #return score.item(), observed_pred
    return observed_pred

# SAVE AND LOAD INITIAL MODELS

def saveInitModels(model_Bald, likelihood_Bald, model_Random, likelihood_Random):
    PATH_Bald = 'static/model/init_state_dict_model_bald.pt'
    PATH_ll_Bald = 'static/model/init_state_dict_ll_bald.pt'
    PATH_Random = 'static/model/init_state_dict_model_random.pt'
    PATH_ll_Random = 'static/model/init_state_dict_ll_random.pt'
    pre_acquisition_model_state_Bald = model_Bald.state_dict()
    pre_acquisition_model_state_Random = model_Random.state_dict()
    pre_acquisition_ll_state_Bald = likelihood_Bald.state_dict()
    pre_acquisition_ll_state_Random = likelihood_Random.state_dict()
    torch.save(pre_acquisition_model_state_Bald, PATH_Bald)
    torch.save(pre_acquisition_ll_state_Bald, PATH_ll_Bald)
    torch.save(pre_acquisition_model_state_Random, PATH_Random)
    torch.save(pre_acquisition_ll_state_Random, PATH_ll_Random)

def loadInitModels(model_Bald, X_train_Bald, likelihood_Bald, model_Random, likelihood_Random):
    PATH_Bald = 'static/model/init_state_dict_model_bald.pt'
    PATH_ll_Bald = 'static/model/init_state_dict_ll_bald.pt'
    PATH_Random = 'static/model/init_state_dict_model_random.pt'
    PATH_ll_Random = 'static/model/init_state_dict_ll_random.pt'
    model_Bald = GPClassificationModel(X_train_Bald)
    model_Bald.load_state_dict(torch.load(PATH_Bald))
    likelihood_Bald = BernoulliLikelihood()
    likelihood_Bald.load_state_dict(torch.load(PATH_ll_Bald))
    # model_Random = GPClassificationModel(X_train_Random)
    model_Random = GPClassificationModel(X_train_Bald)
    model_Random.load_state_dict(torch.load(PATH_Random))
    likelihood_Random = BernoulliLikelihood()
    likelihood_Random.load_state_dict(torch.load(PATH_ll_Random))


# TODO INITIALIZE TRAINING DATA: ADD GUESS AND LAPSE RATE

X_train_1 = torch.linspace(1, 6, 6) # 10, 10)
'''
yTrain_1 = PF_test_function(X_train_1)
yTrainmean_1 = torch.mean(yTrain_1)
y_train_1 = torch.sign(yTrain_1 - yTrainmean_1).add(1).div(2)
for n, lowdelay in enumerate(y_train_1):
    if np.random.uniform(0, 1) <= GAMMA:
        y_train_1[n] = 1
'''
#y_train_1 = (0.51 - 0.49) * torch.rand(5) + 0.49
y_train_1 = torch.Tensor([0, 1, 0, 1, 0, 1])
X_train_2 = torch.linspace(60, 100, 41)
#y_train_2 = (0.99 - 0.95) * torch.rand(41) + 0.95
yTrain_2 = PF_test_function(X_train_2)
yTrainmean_2 = torch.mean(yTrain_2)
y_train_2 = torch.sign(yTrain_2 - yTrainmean_2).add(1).div(2)
for n, highdelay in enumerate(y_train_2):
    if np.random.uniform(0, 1) <= DELTA:
        y_train_2[n] = 0
X_train_Bald = torch.cat((X_train_1, X_train_2))
#del X_train_1
#del X_train_2
# X_train_Random = torch.cat((X_train_1, X_train_2))
y_train = torch.cat((y_train_1, y_train_2))
#del y_train_1
#del y_train_2
# init_trainData_Bald = customDataset(X_train_Bald, y_train)
trainData_Bald = customDataset(X_train_Bald, y_train)
#del y_train
# init_trainData_Random = customDataset(X_train_Random, y_train)
# trainData_Random = customDataset(X_train_Random, y_train)
# trainData_Random = customDataset(X_train_Bald, y_train)

# TODO INITIALIZE TEST DATA: ADD GUESS AND LAPSE RATE

X_test = torch.linspace(1, 100, 100)
yTest = PF_test_function(X_test)
yTestmean = torch.mean(yTest)
y_test = torch.sign(yTest - yTestmean).add(1).div(2)
#del yTestmean
#del yTest
testData_Bald = customDataset(X_test, y_test)
#del X_test
#del y_test
# testData_Random = customDataset(X_test, y_test)

# INITIALIZE POOL DATA
poolData_Bald = torch.linspace(1, 70, 70) #100, 100)
# poolData_Random = X_pool

#test_scores_Bald = []
queried_samples_Bald = []
labels_Bald = []
#test_scores_Random = []
queried_samples_Random = []
labels_Random = []

# INITIALIZE MODELS

model_Bald = GPClassificationModel(X_train_Bald)
likelihood_Bald = BernoulliLikelihood()

# model_Random = GPClassificationModel(X_train_Random)
model_Random = GPClassificationModel(X_train_Bald)
likelihood_Random = BernoulliLikelihood()

#del X_train_Bald

# INITIALIZE ML PARAMETERS

lr = 0.1
training_iterations = 100 #100

# Use the adam optimizer
optimizer_init_Bald = Adam(model_Bald.parameters(), lr=lr)
optimizer_init_Random = Adam(model_Random.parameters(), lr=lr)

# "Loss" for GPs - the marginal log likelihood
mll_init_Bald = VariationalELBO(likelihood_Bald, model_Bald, trainData_Bald.labels.numel())
# mll_init_Random = VariationalELBO(likelihood_Random, model_Random, trainData_Random.labels.numel())
mll_init_Random = VariationalELBO(likelihood_Random, model_Random, trainData_Bald.labels.numel())


# INITIALIZE 2I-2AFC

twoafc = twoafc()

# INITIALIZE STIMULI

stimulus = stimulus()

# INITIALIZE TOTAL COUNTERS
al_counter = 1 # 20, 25, 40
twoafc_counter = 1 #6

# INITIAL TRAINING

train(model=model_Bald, likelihood=likelihood_Bald, optimizer=optimizer_init_Bald, 
    training_iterations=training_iterations, train_data=trainData_Bald, mll=mll_init_Bald)
#train(model=model_Random, likelihood=likelihood_Random, optimizer=optimizer_init_Random, training_iterations=training_iterations, train_data=trainData_Random, mll=mll_init_Random)
train(model=model_Random, likelihood=likelihood_Random, optimizer=optimizer_init_Random, 
    training_iterations=training_iterations, train_data=trainData_Bald, mll=mll_init_Random)
pred_prob_Bald = test(model_Bald, likelihood_Bald, 
    test_data=testData_Bald, criterion=RMSELoss)
# score_Random, pred_prob_Random = test(model_Random, likelihood_Random, test_data=testData_Random, criterion=RMSELoss)
pred_prob_Random = test(model_Random, likelihood_Random, 
    test_data=testData_Bald, criterion=RMSELoss)
#del optimizer_init_Bald
#del mll_init_Bald
#del optimizer_init_Random
#del mll_init_Random

'''
pre_acquisition_model_state_Bald = model_Bald.state_dict()
pre_acquisition_model_state_Random = model_Random.state_dict()
pre_acquisition_ll_state_Bald = likelihood_Bald.state_dict()
pre_acquisition_ll_state_Random = likelihood_Random.state_dict()

PATH_Bald = 'static/model/init_state_dict_model_bald.pt'
PATH_ll_Bald = 'static/model/init_state_dict_ll_bald.pt'
PATH_Random = 'static/model/init_state_dict_model_random.pt'
PATH_ll_Random = 'static/model/init_state_dict_ll_random.pt'
'''

# saveInitModels(PATH_Bald, PATH_ll_Bald, PATH_Random, PATH_ll_Random)

@app.route('/', methods =["POST", "GET"])
def index():
    #name = ""
    #surname = ""
    #session['firstname'] = name
    #session['surname'] = surname
    name = session.get('firstname', None)
    if name:
        name = str(name)
    else:
        name = ''
    surname = session.get('surname', None)
    if surname:
        surname = str(surname)
    else:
        surname = ''
    session['done_Bald'] = False
    session['done_2afc'] = False
    session['done_Rand'] = False
    if request.method == "POST":
        name = str(request.values.get('name'))
        surname = str(request.values.get('lastname'))
        session['firstname'] = name
        session['surname'] = surname
    #silentremove('static/figures/' + name + '_' + surname + '_' + 'PF_BALD_Approximation.png')
    #silentremove('static/figures/' + name + '_' + surname + '_' + 'PF_Random_Approximation.png')
    #silentremove('static/figures/' + name + '_' + surname + '_' + 'PF_WH_Approximation.png')
    silentremove('static/csvs/' + name + '_' + surname + '_results.csv')
    #silentremove('static/csvs/' + name + '_' + surname + '_bald_results.csv')
    #silentremove('static/csvs/' + name + '_' + surname + '_random_results.csv')
    return render_template("index.html")

@app.route('/test_select')
def test_select():
    global queried_samples_Bald
    #global test_scores_Bald
    global labels_Bald
    queried_samples_Bald = []
    #test_scores_Bald = []
    labels_Bald = []
    global queried_samples_Random
    #global test_scores_Random
    global labels_Random
    queried_samples_Random = []
    #test_scores_Random = []
    labels_Random = []
    global testData_Bald
    # TODO CHECK IF/WHEN LOAD INIT MODELS
    # loadInitModels(PATH_Bald, PATH_ll_Bald, PATH_Random, PATH_ll_Random)
    done_Bald = session.get('done_Bald', None)
    done_Rand = session.get('done_Rand', None)
    done_2afc = session.get('done_2afc', None)
    threshold_Bald = session.get('threshold_Bald')
    threshold_Rand = session.get('threshold_Rand')
    threshold_2afc = session.get('threshold_2afc')
    if threshold_Bald:
        threshold_Bald = int(threshold_Bald)
    if threshold_Rand:
        threshold_Rand = int(threshold_Rand)
    if threshold_2afc:
        threshold_2afc = int(threshold_2afc)
    if done_2afc and done_Bald and done_Rand:
        '''
        for n, d in enumerate(data):
            csvname = 'static/csvs/' + name + '_' + surname + '_' + csvString[n] + '_results.csv'
            with open(csvname, 'w') as output_file:
                dict_writer = csv.writer(output_file)
                for key, value in d.items():
                    dict_writer.writerow([key, value])
        '''
        name = str(session.get('firstname', None))
        surname = str(session.get('surname', None))
        queried_Bald = session.get('queried_Bald', None)
        labels_Baldd = session.get('labels_Bald', None)
        #test_data_Bald = session.get('test_data_Bald', None)
        pred_Bald = session.get('pred_Bald', None)
        queried_Rand = session.get('queried_Rand', None)
        labels_Rand = session.get('labels_Rand', None)
        #test_data_Rand = session.get('test_data_Rand', None)
        pred_Rand = session.get('pred_Rand', None)
        queried_2afc = session.get('queried_2afc', None)
        labels_2afc = session.get('labels_2afc', None)
        #test_data_2afc = session.get('test_data_2afc', None)
        pred_2afc = session.get('pred_2afc', None)
        test_csv = testData_Bald.inputs.tolist()
        afc_dict = {'type': '2I-2AFC', 'itds': queried_2afc, 'labels': labels_2afc, 'test': test_csv, 'pred': pred_2afc}
        bald_dict = {'type': 'AL-BALD', 'itds': queried_Bald, 'labels': labels_Baldd, 'test': test_csv, 'pred': pred_Bald} 
        rand_dict = {'type': 'AL-RANDOM', 'itds': queried_Rand, 'labels': labels_Rand, 'test': test_csv, 'pred': pred_Rand}
        data = [afc_dict, bald_dict, rand_dict]
        csvname = 'static/csvs/' + name + '_' + surname + '_results.csv'
        with open(csvname, 'w') as output_file:
            dict_writer = csv.writer(output_file)
            for d in data:
                for key, value in d.items():
                    dict_writer.writerow([key, value])
    return render_template('test_select.html', threshold_Bald=threshold_Bald, threshold_Rand=threshold_Rand, threshold_2afc=threshold_2afc)


@app.route('/test_bald', methods =["POST", "GET"])
def test_bald(model_Bald=model_Bald, likelihood_Bald=likelihood_Bald, 
    pool=poolData_Bald, queried=queried_samples_Bald, labels=labels_Bald,
    traind=trainData_Bald, train_data_new=trainData_Bald):
    answer = 0
    trials = 0
    wavfile = None
    rightmost = 0
    name = str(session.get('firstname', None))
    surname = str(session.get('surname', None))
    if name == 'None':
        name = ''
    if surname == 'None':
        surname = ''
    #pool = poolData_Bald
    #queried = queried_samples_Bald
    #labels = labels_Bald
    #scores = test_scores_Bald
    #traind = trainData_Bald
    #train_data_new = trainData_Bald
    if request.method == "POST":
        # RECEIVE PLAY AND ANSWER
        answer = int(request.values.get('answer'))
        trials = int(request.values.get('trials'))
        if request.values.getlist('poolData_Bald'):
            # RECEIVE AND BUILD TRAIN DATA
            X_traind = torch.Tensor(list(map(float, request.values.getlist('X_train_Bald'))))
            y_traind = torch.Tensor(list(map(float, request.values.getlist('y_train_Bald'))))
            traind = customDataset(X_traind, y_traind)
            # RECEIVE POOL ITDS
            pool = torch.Tensor(list(map(float, request.values.getlist('poolData_Bald'))))
            # RECEIVE LIST OF SCORES, ITDS AND LABELS
            #scores = list(map(float, request.values.getlist('test_scores_Bald')))
            queried = list(map(float, request.values.getlist('queried_samples_Bald')))
            labels = list(map(float, request.values.getlist('labels_Bald')))
        acquirer = BALD(pool.numel())
        best_sample = acquirer.select_samples(model_Bald, likelihood_Bald, pool)
        if answer == 0:
            queried.append(best_sample.item())
            rightmost, wavfile = stimulus.play(best_sample)
            print('ITD queried', best_sample.item())
        else:
            rightmost = int(request.values.get('rightmost'))
            if answer == rightmost:
                label = torch.Tensor([1])
                print('RIGHT! ' + name + ' ' + surname)
            else:
                label = torch.Tensor([0])
                print('WRONG! ' + name + ' ' + surname)
            labels.append(label.item())
            # move that data from the pool to the training set
            pool, train_data_new = move_s(best_sample, label, pool, traind)
            # init the optimizer
            optimizer_Bald = Adam(model_Bald.parameters(), lr=lr)
            # init the marginal likelihood
            mll_Bald = VariationalELBO(likelihood=likelihood_Bald, 
                model=model_Bald, num_data=train_data_new.inputs.numel())
            # re-train the model
            train(model=model_Bald, likelihood=likelihood_Bald, optimizer=optimizer_Bald, 
                training_iterations=training_iterations, train_data=train_data_new, mll=mll_Bald)
            # test the model and compute the score 
            # TODO FIND A STOP CRITERION AND A METRIC FOR SCORE
            pred_prob = test(model=model_Bald, likelihood=likelihood_Bald,
                test_data=testData_Bald, criterion=RMSELoss)
            #scores.append(score)
            #print('score', score)
            '''
            max_var, ind = torch.max(pred_prob.variance, 0)
            # print('MAX Variance', max_var)
            f, ax = plt.subplots(1, 1)
            ax.tick_params(left = False)
            ax.set_ylim(-0.3, 1.3)
            ax.scatter(init_trainData_Bald.inputs.reshape(-1, 1).numpy(), init_trainData_Bald.labels.numpy(),  marker='*')
            ax.scatter(queried, labels,  marker='*', color='b')
            # ax.plot(testData.inputs.numpy(), testData.labels.numpy(), 'b')
            ax.plot(testData_Bald.inputs.numpy(), pred_prob.mean, 'r')
            double_std = torch.sqrt(pred_prob.variance)
            lower = pred_prob.mean - double_std
            upper = pred_prob.mean + double_std
            seventynine_percent = min(pred_prob.mean, key= lambda x: abs(x - 0.794))
            seventy_index = (pred_prob.mean == seventynine_percent).nonzero(as_tuple=True)[0]
            seventies = testData_Bald.inputs[seventy_index]
            ax.fill_between(testData_Bald.inputs.numpy(), lower.numpy(), upper.numpy(), alpha=0.5, color='r')
            ax.legend(['Train Data', 'Latent PF on test data', 'Predicted probabilities' + '\n' + 'Max variance: {:.2f}'.format(max_var.item()) + '\n' + 'at: {:.0f} '.format(testData_Bald.inputs.numpy()[ind]) + r'$\mu$s'])
            ax.set_xlabel('ITD')
            # ax.set_ylabel('Probability')
            # ax.set_title(f'{acquirer.__class__.__name__}' + ' PF Fitting')
            ax.axvline(best_sample.item(), 0, 1)
            ax.set_title('BALD' + ' PF Fitting: 79.4% at {:.0f}'.format(seventies[0].item()) + r'$\mu$s')
            plt.savefig('static/figures/PF_' + f'{acquirer.__class__.__name__}' + '_Approximation_' + str(trials) + '.png')
            plt.close(f)
            '''
        if trials == al_counter:
            # Plot the PF curve
            '''
            f, ax = plt.subplots(1, 1)
            ax.tick_params(left = False)
            ax.set_ylim(-0.3, 1.3)
            ax.scatter(trainData_Bald.inputs.reshape(-1, 1).numpy(), trainData_Bald.labels.numpy(),  
                marker='*')
            ax.scatter(queried, labels,  marker='d', color='b')
            ax.plot(testData_Bald.inputs.numpy(), pred_prob.mean, 'r')
            double_std = torch.sqrt(pred_prob.variance)
            lower = pred_prob.mean - double_std
            upper = pred_prob.mean + double_std
            max_var, ind = torch.max(pred_prob.variance, 0)
            seventynine_percent = min(pred_prob.mean, key= lambda x: abs(x - 0.794))
            seventy_index = (pred_prob.mean == seventynine_percent).nonzero(as_tuple=True)[0]
            seventies = testData_Bald.inputs[seventy_index]
            # print('79.4% point PF curve: ', testData_Bald.inputs[seventy_index])
            ax.fill_between(testData_Bald.inputs.numpy(), lower.numpy(), upper.numpy(), alpha=0.5, 
                color='r')
            ax.legend(['Train Data', 'Latent PF on test data', 'Predicted probabilities' + '\n' + 'Max variance: {:.2f}'.format(max_var.item()) + '\n' + 'at: {:.0f} '.format(testData_Bald.inputs.numpy()[ind]) + r'$\mu$s'])
            ax.set_xlabel('ITD')
            ax.set_title('BALD' + ' PF Fitting: 79.4% at {:.0f}'.format(seventies[0].item()))
            plt.savefig('static/figures/' + name + '_' + surname + '_' + 'PF_BALD_Approximation.png')
            plt.close(f)
            '''
            seventynine_percent = min(pred_prob.mean, key= lambda x: abs(x - 0.794))
            seventy_index = (pred_prob.mean == seventynine_percent).nonzero(as_tuple=True)[0]
            seventies = testData_Bald.inputs[seventy_index]
            session['threshold_Bald'] = seventies[0].item()
            session['queried_Bald'] = queried
            session['labels_Bald'] = labels
            #session['test_data_Bald'] = testData_Bald.inputs.tolist()
            session['pred_Bald'] = pred_prob.mean.tolist()
            session['done_Bald'] = True
            #del model_Bald
            #del likelihood_Bald
            #del optimizer_Bald
            #del mll_Bald
        return {'wav_location': wavfile, 'itd': best_sample.item(), 'rightmost': rightmost,
            'Xtrain': train_data_new.inputs.tolist(), 'ytrain': train_data_new.labels.tolist(), 
            'pooldata': pool.tolist(), 'trials': trials,
            'queries': queried, 'labels': labels}
    return render_template('test_bald.html')


@app.route('/test_random', methods =["POST", "GET"])
def test_random(model_Random=model_Random, likelihood_Random=likelihood_Random, 
    pool=poolData_Bald, queried=queried_samples_Random, labels=labels_Random,
    traind=trainData_Bald, train_data_new=trainData_Bald):
    trials = 0
    answer = 0
    wavfile = None
    rightmost = 0
    name = str(session.get('firstname', None))
    surname = str(session.get('surname', None))
    if name == 'None':
        name = ''
    if surname == 'None':
        surname = ''
    # pool = poolData_Random
    #pool = poolData_Bald
    #queried = queried_samples_Random
    #labels = labels_Random
    #scores = test_scores_Random
    # traind = trainData_Random
    # train_data_new = trainData_Random
    #traind = trainData_Bald
    #train_data_new = trainData_Bald
    if request.method == "POST":
        # RECEIVE PLAY AND ANSWER
        answer = int(request.values.get('answer'))
        trials = int(request.values.get('trials'))
        if request.values.getlist('queried_samples_Random'):
            # RECEIVE AND BUILD TRAIN DATA
            X_traind = torch.Tensor(list(map(float, request.values.getlist('X_train_Random'))))
            y_traind = torch.Tensor(list(map(float, request.values.getlist('y_train_Random'))))
            traind = customDataset(X_traind, y_traind)
            # RECEIVE POOL ITDS
            pool = torch.Tensor(list(map(float, request.values.getlist('poolData_Random'))))
            # RECEIVE LIST OF SCORES, ITDS AND LABELS
            #scores = list(map(float, request.values.getlist('test_scores_Random')))
            queried = list(map(float, request.values.getlist('queried_samples_Random')))
            labels = list(map(float, request.values.getlist('labels_Random')))
        acquirer = Random(pool.numel())
        best_sample = acquirer.select_samples(model_Random, likelihood_Random, pool)
        if answer == 0:
            queried.append(best_sample.item())
            rightmost, wavfile = stimulus.play(best_sample)
            print('ITD queried', best_sample.item())
        else:
            rightmost = int(request.values.get('rightmost'))
            # print('rightmost', rightmost)
            if answer == rightmost:
                label = torch.Tensor([1])
                print('RIGHT! ' + name + ' ' + surname)
            else:
                label = torch.Tensor([0])
                print('WRONG! ' + name + ' ' + surname)
            labels.append(label.item())
            # move that data from the pool to the training set
            pool, train_data_new = move_s(best_sample, label, pool, traind)
            # init the optimizer
            optimizer_Random = Adam(model_Random.parameters(), lr=lr)
            # init the marginal likelihood
            mll_Random = VariationalELBO(likelihood=likelihood_Random, 
                model=model_Random, num_data=train_data_new.inputs.numel())
            # re-train the model
            train(model=model_Random, likelihood=likelihood_Random, optimizer=optimizer_Random, 
                training_iterations=training_iterations, train_data=train_data_new, mll=mll_Random)
            # test the model and compute the score 
            # TODO FIND A STOP CRITERION AND A METRIC FOR SCORE
            # score, pred_prob = test(model=model_Random, likelihood=likelihood_Random, test_data=testData_Random, criterion=RMSELoss)
            pred_prob = test(model=model_Random, likelihood=likelihood_Random, 
                test_data=testData_Bald, criterion=RMSELoss)
            #test_scores_Random.append(score)
            '''
            max_var, ind = torch.max(pred_prob.variance, 0)
            # print('MAX Variance', max_var)
            f, ax = plt.subplots(1, 1)
            ax.tick_params(left = False)
            ax.set_ylim(-0.3, 1.3)
            ax.scatter(init_trainData_Random.inputs.reshape(-1, 1).numpy(), init_trainData_Random.labels.numpy(),  marker='*')
            ax.scatter(queried, labels,  marker='*', color='b')
            # ax.plot(testData.inputs.numpy(), testData.labels.numpy(), 'b')
            ax.plot(testData_Random.inputs.numpy(), pred_prob.mean, 'r')
            double_std = torch.sqrt(pred_prob.variance)
            lower = pred_prob.mean - double_std
            upper = pred_prob.mean + double_std
            seventynine_percent = min(pred_prob.mean, key= lambda x: abs(x - 0.794))
            seventy_index = (pred_prob.mean == seventynine_percent).nonzero(as_tuple=True)[0]
            seventies = testData_Random.inputs[seventy_index]
            ax.fill_between(testData_Random.inputs.numpy(), lower.numpy(), upper.numpy(), alpha=0.5, color='r')
            ax.legend(['Train Data', 'Latent PF on test data', 'Predicted probabilities' + '\n' + 'Max variance: {:.2f}'.format(max_var.item()) + '\n' + 'at: {:.0f} '.format(testData_Random.inputs.numpy()[ind]) + r'$\mu$s'])
            ax.set_xlabel('ITD')
            # ax.set_ylabel('Probability')
            # ax.set_title(f'{acquirer.__class__.__name__}' + ' PF Fitting')
            ax.axvline(best_sample.item(), 0, 1)
            ax.set_title('Random' + ' PF Fitting: 79.4% at {:.0f}'.format(seventies[0].item()) + r'$\mu$s')
            plt.savefig('static/figures/PF_' + f'{acquirer.__class__.__name__}' + '_Approximation_' + str(trials) + '.png')
            plt.close(f)
            '''
            
        if trials == al_counter:
            # Plot the PF curve
            '''
            f, ax = plt.subplots(1, 1)
            ax.tick_params(left = False)
            ax.set_ylim(-0.3, 1.3)
            # ax.scatter(trainData_Random.inputs.reshape(-1, 1).numpy(), trainData_Random.labels.numpy(), marker='*')
            ax.scatter(trainData_Bald.inputs.reshape(-1, 1).numpy(), trainData_Bald.labels.numpy(), marker='*')
            ax.scatter(queried, labels,  marker='d', color='b')
            # ax.plot(testData_Random.inputs.numpy(), pred_prob.mean, 'r')
            ax.plot(testData_Bald.inputs.numpy(), pred_prob.mean, 'r')
            double_std = torch.sqrt(pred_prob.variance)
            lower = pred_prob.mean - double_std
            upper = pred_prob.mean + double_std
            max_var, ind = torch.max(pred_prob.variance, 0)
            seventynine_percent = min(pred_prob.mean, key= lambda x: abs(x - 0.794))
            seventy_index = (pred_prob.mean == seventynine_percent).nonzero(as_tuple=True)[0]
            # print('79.4% point PF curve: ', testData_Bald.inputs[seventy_index])
            # seventies = testData_Random.inputs[seventy_index]
            seventies = testData_Bald.inputs[seventy_index]
            # ax.fill_between(testData_Random.inputs.numpy(), lower.numpy(), upper.numpy(), alpha=0.5, color='r')
            ax.fill_between(testData_Bald.inputs.numpy(), lower.numpy(), upper.numpy(), 
                alpha=0.5, color='r')
            ax.legend(['Train Data', 'Latent PF on test data', 'Predicted probabilities' + '\n' + 'Max variance: {:.2f}'.format(max_var.item()) + '\n' + 'at: {:.0f} '.format(testData_Random.inputs.numpy()[ind]) + r'$\mu$s'])
            ax.set_xlabel('ITD')
            ax.set_title('Random' + ' PF Fitting: 79.4% at {:.0f}'.format(seventies[0].item()))
            plt.savefig('static/figures/' + name + '_' + surname + '_' + 'PF_Random_Approximation.png')
            plt.close(f)
            '''
            seventynine_percent = min(pred_prob.mean, key= lambda x: abs(x - 0.794))
            seventy_index = (pred_prob.mean == seventynine_percent).nonzero(as_tuple=True)[0]
            seventies = testData_Bald.inputs[seventy_index]
            session['threshold_Rand'] = seventies[0].item()
            session['queried_Rand'] = queried
            session['labels_Rand'] = labels
            # session['test_data_Rand'] = testData_Bald.inputs.tolist()
            session['pred_Rand'] = pred_prob.mean.tolist()
            session['done_Rand'] = True
            #del model_Random
            #del likelihood_Random
            #del optimizer_Random
            #del mll_Random
        return {'wav_location': wavfile, 'itd': best_sample.item(), 'rightmost': rightmost,
            'Xtrain': train_data_new.inputs.tolist(), 'ytrain': train_data_new.labels.tolist(), 
            'pooldata': pool.tolist(), 'trials': trials,
            'queries': queried, 'labels': labels}
    return render_template('test_random.html')


@app.route('/test_2afc', methods =["POST", "GET"])
def test_2afc():
    name = str(session.get('firstname', None))
    surname = str(session.get('surname', None))
    if name == 'None':
        name = ''
    if surname == 'None':
        surname = ''
    answer = 0
    wavfile = None
    queried = []
    labels = []
    # INITALIZE TRIAL COUNTER
    # Counting the trials
    counter = 0
    # Counting consecutive right answers
    correct_counter = 0
    # Flags of last ITD transformation
    upsized = 0
    downsized = 0
    # Starting ITD and step size
    itd = twoafc.start_itd
    factor = twoafc.initial_step
    reversals = twoafc.reversals
    downup_reversals = twoafc.downup_reversals
    if request.method == "POST":
        # while twoafc.reversals < twoafc.total_reversals:
        # play the stimulus
        answer = int(request.values.get('answer'))
        # play = int(request.values.get('ajaxPlay'))
        if request.values.getlist('queried_samples'):
            counter = int(request.values.get('counter'))
            itd = float(request.values.get('itd'))
            factor = float(request.values.get('factor'))
            correct_counter = int(request.values.get('correct_counter'))
            upsized = int(request.values.get('upsized'))
            downsized = int(request.values.get('downsized'))
            reversals = int(request.values.get('reversals'))
            downup_reversals = int(request.values.get('downup_reversals'))
            queried = list(map(float, request.values.getlist('queried_samples')))
            labels = list(map(float, request.values.getlist('labels')))
        if answer == 0:
            queried.append(itd)
            rightmost, wavfile = stimulus.play(itd)
            print('ITD queried', itd)
        else:
            rightmost = int(request.values.get('rightmost'))
            if answer == rightmost:
                label = 1
                correct_counter += 1
                print('RIGHT! ' + name + ' ' + surname)
            else:
                label = 0
                correct_counter = 0
                print('WRONG! ' + name + ' ' + surname)
            # update counters and answers dictionary
            counter += 1
            labels.append(label)
            # first two tests: wrong -> up, right -> same
            # WHY!?!?
            if counter <= 2:
                if label == 0:
                    itd = factor * itd
                    upsized = 1
            else:
                # update factor wrt downup reversals
                if downup_reversals == 0:
                    factor = twoafc.initial_step
                elif downup_reversals == 1:
                    factor = twoafc.first_downup_step
                else:
                    factor = twoafc.second_downup_step
                # three times right in a row -> down
                if correct_counter == 3:
                    itd = itd / factor
                    downsized = 1
                    # if up down, increment reversals counter and unflag up
                    if upsized:
                        reversals += 1
                        upsized = 0
                    correct_counter = 0
                # wrong -> up
                if label == 0:
                    itd = factor * itd
                    upsized = 1
                    # if down up, increment reversals counter 
                    # and downup reversals counter, unflag down
                    if downsized:
                        reversals += 1
                        downup_reversals += 1
                        downsized = 0
            # count reversals only after the minimum step size
            if downup_reversals < 2:
                reversals = 0
        if reversals == twoafc_counter: #twoafc.total_reversals:
            itds = np.asarray(queried)
            labels_array = np.asarray(labels)
            inds = itds.argsort()
            labels_sorted = labels_array[inds]
            itds_sorted = itds[inds]
            pc = PsychometricCurve(model='wh').fit(itds_sorted, labels_sorted)
            unique_itds = np.unique(itds_sorted)
            predictions = pc.predict(unique_itds)
            seventynine_percent = min(predictions, key= lambda x: abs(x - 0.794))
            seventy_index = (predictions == seventynine_percent).nonzero()[0].item()
            # pc.plot(itds_sorted, labels_sorted, name, surname, unique_itds[seventy_index].item())
            session['threshold_2afc'] = unique_itds[seventy_index].item()
            session['queried_2afc'] = queried
            session['labels_2afc'] = labels
            # session['test_data_2afc'] = testData_Bald.inputs.tolist()
            session['pred_2afc'] = pc.predict(testData_Bald.inputs.numpy()).tolist()
            session['done_2afc'] = True
        return {'wav_location': wavfile, 'itd': itd, 'factor': factor,
            'counter': counter, 'correct_counter': correct_counter, 
            'upsized': upsized, 'downsized': downsized, 'rightmost': rightmost,
            'reversals': reversals, 'downup_reversals': downup_reversals, 
            'queries': queried, 'labels': labels}
    return render_template('test_2afc.html')

if __name__== '__main__':
    app.config['SESSION_TYPE'] = 'filesystem'
    #Session(app)
    print('APP IS MAIN!')
    app.run(debug=True)
