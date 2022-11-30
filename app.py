# IMPORTS

from flask import Flask, request, session, render_template
# from flask_session.__init__ import Session
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import VariationalELBO

from matplotlib import pyplot as plt
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import torch
import gpytorch
from torch.optim import Adam
import secrets
import time
import os
import errno
import sys
#import io
#import base64
import numpy as np
# from copy import deepcopy
sys.path.insert(0, os.getcwd() + '/modules')  
from customDataset import CustomDataset as customDataset
from sound import Stimulus as stimulus
from acquisition import BALD as BALD
from acquisition import Random as Random
# from util import move_sample as move_sample
from util import move_s
from util import RMSELoss
from twoAFC import TwoAFC as twoafc
from psychometric_curve import PsychometricCurve

# INITIALIZE FLASK APP
secret = secrets.token_urlsafe(32)
app = Flask(__name__)
app.secret_key = secret
torch.set_flush_denormal(True)
plt.switch_backend('Agg')

# CLASSES AND METHODS

# REMOVE FILES
def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred

# TODO DEFINE PSYCHOMETRIC LATENT FUNCTION

ALPHA = 25 # threshold
BETA = 10 # slope
GAMMA = 0.5 # guess rate
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
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred

# TRAIN AND TEST METHODS

def train(model, likelihood, optimizer, training_iterations, train_data, mll):
    startTrain = time.time()
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
    endTrain = time.time()
    # REMOVE
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
        pred_labels = observed_pred.mean.ge(0.5).float()
        score = criterion(pred_labels, test_data.labels)
        # print('RMSE SCORE: ', score)
    # endTest = time.time()
    # print('TEST TIME', endTest - startTest)
    return score.item(), observed_pred

# SAVE AND LOAD INITIAL MODELS

def saveInitModels(pathBald, pathllBald, pathRandom, pathllRandom):
    torch.save(pre_acquisition_model_state_Bald, pathBald)
    torch.save(pre_acquisition_model_state_Bald, pathllBald)
    torch.save(pre_acquisition_model_state_Random, pathRandom)
    torch.save(pre_acquisition_ll_state_Random, pathllRandom)

def loadInitModels(pathBald, pathllBald, pathRandom, pathllRandom):
    global model_Bald
    global likelihood_Bald
    global model_Random
    global likelihood_Random
    model_Bald = torch.load(pathBald)
    likelihood_Bald = torch.load(pathllBald)
    model_Random = torch.load(pathRandom)
    likelihood_Random = torch.load(pathllRandom)


# TODO INITIALIZE TRAINING DATA: ADD GUESS AND LAPSE RATE

X_train_1 = torch.linspace(1, 10, 10)
yTrain_1 = PF_test_function(X_train_1)
yTrainmean_1 = torch.mean(yTrain_1)
y_train_1 = torch.sign(yTrain_1 - yTrainmean_1).add(1).div(2)
#for n, lowdelay in enumerate(y_train_1):
#    if np.random.uniform(0, 1) <= GAMMA:
#        y_train_1[n] = 1
X_train_2 = torch.linspace(60, 100, 41)
yTrain_2 = PF_test_function(X_train_2)
yTrainmean_2 = torch.mean(yTrain_2)
y_train_2 = torch.sign(yTrain_2 - yTrainmean_2).add(1).div(2)
#for n, highdelay in enumerate(y_train_2):
#    if np.random.uniform(0, 1) <= DELTA:
#        y_train_2[n] = 0
X_train_Bald = torch.cat((X_train_1, X_train_2))
X_train_Random = torch.cat((X_train_1, X_train_2))
y_train = torch.cat((y_train_1, y_train_2))
init_trainData_Bald = customDataset(X_train_Bald, y_train)
trainData_Bald = customDataset(X_train_Bald, y_train)
init_trainData_Random = customDataset(X_train_Random, y_train)
trainData_Random = customDataset(X_train_Random, y_train)

# TODO INITIALIZE TEST DATA: ADD GUESS AND LAPSE RATE

X_test = torch.linspace(1, 100, 100)
yTest = PF_test_function(X_test)
yTestmean = torch.mean(yTest)
y_test = torch.sign(yTest - yTestmean).add(1).div(2)
testData_Bald = customDataset(X_test, y_test)
testData_Random = customDataset(X_test, y_test)

# INITIALIZE MODELS

model_Bald = GPClassificationModel(X_train_Bald)
likelihood_Bald = BernoulliLikelihood()

model_Random = GPClassificationModel(X_train_Random)
likelihood_Random = BernoulliLikelihood()

# INITIALIZE ML PARAMETERS

lr = 0.1
training_iterations = 50

# Use the adam optimizer
optimizer_init_Bald = Adam(model_Bald.parameters(), lr=lr)
optimizer_init_Random = Adam(model_Random.parameters(), lr=lr)

# "Loss" for GPs - the marginal log likelihood
mll_init_Bald = VariationalELBO(likelihood_Bald, model_Bald, trainData_Bald.labels.numel())
mll_init_Random = VariationalELBO(likelihood_Random, model_Random, trainData_Random.labels.numel())

# INITIALIZE 2I-2AFC

twoafc = twoafc()

# INITIALIZE STIMULI

stimulus = stimulus()

# INITIALIZE TOTAL COUNTERS
al_counter = 4 # 40
twoafc_counter = 2 #6

# INITIAL TRAINING

train(model=model_Bald, likelihood=likelihood_Bald, optimizer=optimizer_init_Bald, 
    training_iterations=training_iterations, train_data=trainData_Bald, mll=mll_init_Bald)
train(model=model_Random, likelihood=likelihood_Random, optimizer=optimizer_init_Random, 
    training_iterations=training_iterations, train_data=trainData_Random, mll=mll_init_Random)
score_Bald, pred_prob_Bald = test(model_Bald, likelihood_Bald, test_data=testData_Bald, criterion=RMSELoss)
score_Random, pred_prob_Random = test(model_Random, likelihood_Random, test_data=testData_Random, criterion=RMSELoss)

pre_acquisition_model_state_Bald = model_Bald.state_dict()
pre_acquisition_model_state_Random = model_Random.state_dict()
pre_acquisition_ll_state_Bald = likelihood_Bald.state_dict()
pre_acquisition_ll_state_Random = likelihood_Random.state_dict()

# INITIALIZE POOL DATA
X_pool = torch.linspace(1, 100, 100)
yPool = PF_test_function(X_pool)
yPoolmean = torch.mean(yPool)
y_pool = torch.sign(yPool - yPoolmean).add(1).div(2)
poolData_Bald = X_pool
poolData_Random = X_pool

test_scores_Bald = []
queried_samples_Bald = []
labels_Bald = []
test_scores_Random = []
queried_samples_Random = []
labels_Random = []

PATH_Bald = 'static/model/init_state_dict_model_bald.pt'
PATH_ll_Bald = 'static/model/init_state_dict_ll_bald.pt'
PATH_Random = 'static/model/init_state_dict_model_random.pt'
PATH_ll_Random = 'static/model/init_state_dict_ll_random.pt'

# saveInitModels(PATH_Bald, PATH_ll_Bald, PATH_Random, PATH_ll_Random)

@app.route('/', methods =["POST", "GET"])
def index():
    name = ""
    surname = ""
    #session['firstname'] = name
    #session['surname'] = surname
    if request.method == "POST":
        name = str(request.values.get('name'))
        surname = str(request.values.get('lastname'))
        #session['firstname'] = name
        #session['surname'] = surname
        # REMOVE
    #print('firstname', session['firstname'])
    #print('surname', surname)
    #silentremove('static/figures/' + name + '_' + surname + '_' + 'PF_BALD_Approximation.png')
    #silentremove('static/figures/' + name + '_' + surname + '_' + 'PF_Random_Approximation.png')
    #silentremove('static/figures/' + name + '_' + surname + '_' + 'PF_WH_Approximation.png')
    return render_template("index.html")

@app.route('/test_select')
def test_select():
    global queried_samples_Bald
    global test_scores_Bald
    global labels_Bald
    queried_samples_Bald = []
    test_scores_Bald = []
    labels_Bald = []
    global queried_samples_Random
    global test_scores_Random
    global labels_Random
    test_scores_Random = []
    labels_Random = []
    queried_samples_Random = []
    #imgBald = session.get('imageBald', None)
    # TODO CHECK WHERE TO LOAD INIT MODELS
    #loadInitModels(PATH_Bald, PATH_ll_Bald, PATH_Random, PATH_ll_Random)
    return render_template('test_select.html') #, imgBald = imgBald)


@app.route('/test_bald', methods =["POST", "GET"])
def test_bald():
    answer = 0
    trials = 0
    wavfile = None
    rightmost = 0
    name = str(session.get('firstname', None))
    surname = str(session.get('surname', None))
    pool = poolData_Bald
    queried = queried_samples_Bald
    labels = labels_Bald
    scores = test_scores_Bald
    traind = trainData_Bald
    if request.method == "POST":
        # RECEIVE PLAY AND ANSWER
        answer = int(request.values.get('answer'))
        trials = int(request.values.get('trials'))
        #play = int(request.values.get('ajaxPlay'))
        if request.values.getlist('queried_samples_Bald'):
            # RECEIVE AND BUILD TRAIN DATA
            X_traind = torch.Tensor(list(map(float, request.values.getlist('X_train_Bald'))))
            y_traind = torch.Tensor(list(map(float, request.values.getlist('y_train_Bald'))))
            traind = customDataset(X_traind, y_traind)
            # RECEIVE POOL ITDS
            pool = torch.Tensor(list(map(float, request.values.getlist('poolData_Bald'))))
            # RECEIVE LIST OF SCORES, ITDS AND LABELS
            scores = list(map(float, request.values.getlist('test_scores_Bald')))
            queried = list(map(float, request.values.getlist('queried_samples_Bald')))
            labels = list(map(float, request.values.getlist('labels_Bald')))
        acquirer = BALD(pool.numel())
        best_sample = acquirer.select_samples(model_Bald, likelihood_Bald, pool)
        # print('ITD', best_sample.item())
        # print('answer', answer)
        if answer == 0:
            queried.append(best_sample.item())
            rightmost, wavfile = stimulus.play(best_sample)
            # print('rightmost', rightmost)
        else:
            rightmost = int(request.values.get('rightmost'))
            # print('rightmost', rightmost)
            if answer == rightmost:
                label = torch.Tensor([1])
                print('RIGHT! and name ' + name + surname)
            else:
                label = torch.Tensor([0])
                print('WRONG! and name' + name + surname)
            labels.append(label.item())
            # move that data from the pool to the training set
            pool = move_s(best_sample, label, pool, traind)

            # init the optimizer
            optimizer_Bald = Adam(model_Bald.parameters(), lr=lr)

            # init the marginal likelihood
            mll_Bald = VariationalELBO(likelihood=likelihood_Bald, 
                model=model_Bald, num_data=traind.inputs.numel())
            
            # re-train the model
            train(model=model_Bald, likelihood=likelihood_Bald, optimizer=optimizer_Bald, 
                training_iterations=training_iterations, train_data=traind, mll=mll_Bald)
                
            # test the model and compute the score 
            # TODO FIND A STOP CRITERION AND A METRIC FOR SCORE
            score, pred_prob = test(model=model_Bald, likelihood=likelihood_Bald,
                test_data=testData_Bald, criterion=RMSELoss)
            scores.append(score)
        # print('queried', queried)
        # print('labels', labels)
        if trials == al_counter:
            # Plot the PF curve
            f, ax = plt.subplots(1, 1)
            ax.tick_params(left = False)
            ax.set_ylim(-0.3, 1.3)
            ax.scatter(trainData_Bald.inputs.reshape(-1, 1).numpy(), trainData_Bald.labels.numpy(),  marker='*')
            ax.scatter(queried, labels,  marker='*', color='b')
            ax.plot(testData_Bald.inputs.numpy(), pred_prob.mean, 'r')
            double_std = torch.sqrt(pred_prob.variance)
            lower = pred_prob.mean - double_std
            upper = pred_prob.mean + double_std
            max_var, ind = torch.max(pred_prob.variance, 0)
            seventynine_percent = min(pred_prob.mean, key= lambda x: abs(x - 0.794))
            seventy_index = (pred_prob.mean == seventynine_percent).nonzero(as_tuple=True)[0]
            # print('79.4% point PF curve: ', testData_Bald.inputs[seventy_index])
            ax.fill_between(testData_Bald.inputs.numpy(), lower.numpy(), upper.numpy(), alpha=0.5, color='r')
            ax.legend(['Train Data', 'Latent PF on test data', 'Predicted probabilities' + '\n' + 'Max variance: {:.2f}'.format(max_var.item()) + '\n' + 'at: {:.0f} '.format(testData_Bald.inputs.numpy()[ind]) + r'$\mu$s'])
            ax.set_xlabel('ITD')
            ax.set_title(f'{acquirer.__class__.__name__}' + ' PF Fitting: 79.4% at {:.0f}'.format(testData_Bald.inputs[seventy_index].item()))
            #pngImage = io.BytesIO()
            #FigureCanvas(f).print_png(pngImage)
            #pngImageB64String = "data:image/png;base64,"
            #pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
            #session['imageBald'] = pngImageB64String
            print('saving image')
            plt.savefig('static/figures/' + name + '_' + surname + '_' + 'PF_BALD_Approximation.png')
            plt.close(f)
        return {'wav_location': wavfile, 'itd': best_sample.item(), 'rightmost': rightmost,
            'Xtrain': traind.inputs.tolist(), 'ytrain': traind.labels.tolist(), 
            'pooldata': pool.tolist(), 'scores': scores, 'trials': trials,
            'queries': queried, 'labels': labels}
    return render_template('test_bald.html')


@app.route('/test_random', methods =["POST", "GET"])
def test_random():
    trials = 0
    answer = 0
    wavfile = None
    rightmost = 0
    name = str(session.get('firstname', None))
    surname = str(session.get('surname', None))
    pool = poolData_Random
    queried = queried_samples_Random
    labels = labels_Random
    scores = test_scores_Random
    traind = trainData_Random
    if request.method == "POST":
        # RECEIVE PLAY AND ANSWER
        answer = int(request.values.get('answer'))
        trials = int(request.values.get('trials'))
        #play = int(request.values.get('ajaxPlay'))
        if request.values.getlist('queried_samples_Random'):
            # RECEIVE AND BUILD TRAIN DATA
            X_traind = torch.Tensor(list(map(float, request.values.getlist('X_train_Random'))))
            y_traind = torch.Tensor(list(map(float, request.values.getlist('y_train_Random'))))
            traind = customDataset(X_traind, y_traind)
            # RECEIVE POOL ITDS
            pool = torch.Tensor(list(map(float, request.values.getlist('poolData_Random'))))
            # RECEIVE LIST OF SCORES, ITDS AND LABELS
            scores = list(map(float, request.values.getlist('test_scores_Random')))
            queried = list(map(float, request.values.getlist('queried_samples_Random')))
            labels = list(map(float, request.values.getlist('labels_Random')))
        acquirer = Random(pool.numel())
        best_sample = acquirer.select_samples(model_Random, likelihood_Random, pool)
        # print('ITD', best_sample.item())
        # print('answer', answer)
        if answer == 0:
            queried.append(best_sample.item())
            rightmost, wavfile = stimulus.play(best_sample)
            # print('rightmost', rightmost)
        else:
            rightmost = int(request.values.get('rightmost'))
            # print('rightmost', rightmost)
            if answer == rightmost:
                label = torch.Tensor([1])
                print('RIGHT! and name ' + name + surname)
            else:
                label = torch.Tensor([0])
                print('WRONG! and name  ' + name + surname)
            labels.append(label.item())
            # move that data from the pool to the training set
            pool = move_s(best_sample, label, pool, traind)

            # init the optimizer
            optimizer_Random = Adam(model_Random.parameters(), lr=lr)

            # init the marginal likelihood
            mll_Random = VariationalELBO(likelihood=likelihood_Random, 
                model=model_Random, num_data=traind.inputs.numel())
            
            # re-train the model
            train(model=model_Random, likelihood=likelihood_Random, optimizer=optimizer_Random, 
                training_iterations=training_iterations, train_data=traind, mll=mll_Random)
                
            # test the model and compute the score 
            # TODO FIND A STOP CRITERION AND A METRIC FOR SCORE
            score, pred_prob = test(model=model_Random, likelihood=likelihood_Random,
                test_data=testData_Random, criterion=RMSELoss)
            test_scores_Random.append(score)
        # print('queried', queried)
        # print('labels', labels)
        if trials == al_counter:
            # Plot the PF curve
            f, ax = plt.subplots(1, 1)
            ax.tick_params(left = False)
            ax.set_ylim(-0.3, 1.3)
            ax.scatter(trainData_Random.inputs.reshape(-1, 1).numpy(), trainData_Random.labels.numpy(),  marker='*')
            ax.scatter(queried, labels,  marker='*', color='b')
            ax.plot(testData_Random.inputs.numpy(), pred_prob.mean, 'r')
            double_std = torch.sqrt(pred_prob.variance)
            lower = pred_prob.mean - double_std
            upper = pred_prob.mean + double_std
            max_var, ind = torch.max(pred_prob.variance, 0)
            seventynine_percent = min(pred_prob.mean, key= lambda x: abs(x - 0.794))
            seventy_index = (pred_prob.mean == seventynine_percent).nonzero(as_tuple=True)[0]
            # print('79.4% point PF curve: ', testData_Bald.inputs[seventy_index])
            ax.fill_between(testData_Random.inputs.numpy(), lower.numpy(), upper.numpy(), alpha=0.5, color='r')
            ax.legend(['Train Data', 'Latent PF on test data', 'Predicted probabilities' + '\n' + 'Max variance: {:.2f}'.format(max_var.item()) + '\n' + 'at: {:.0f} '.format(testData_Random.inputs.numpy()[ind]) + r'$\mu$s'])
            ax.set_xlabel('ITD')
            ax.set_title(f'{acquirer.__class__.__name__}' + ' PF Fitting: 79.4% at {:.0f}'.format(testData_Random.inputs[seventy_index].item()))
            print('saving image')
            plt.savefig('static/figures/' + name + '_' + surname + '_' + 'PF_Random_Approximation.png')
            plt.close(f)
        return {'wav_location': wavfile, 'itd': best_sample.item(), 'rightmost': rightmost,
            'Xtrain': traind.inputs.tolist(), 'ytrain': traind.labels.tolist(), 
            'pooldata': pool.tolist(), 'scores': scores, 'trials': trials,
            'queries': queried, 'labels': labels}
    return render_template('test_random.html')


@app.route('/test_2afc', methods =["POST", "GET"])
def test_2afc():
    name = str(session.get('firstname', None))
    surname = str(session.get('surname', None))
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
            # COMPUTE EVERYTHING
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
        # print('ITD', itd)
        # print('answer', answer)
        # print('TOTAL REVERSALS', reversals)
        # print('DOWNUP REVERSALS', downup_reversals)
        if answer == 0:
            queried.append(itd)
            rightmost, wavfile = stimulus.play(itd)
        else:
            rightmost = int(request.values.get('rightmost'))
            if answer == rightmost:
                label = 1
                correct_counter += 1
                print('RIGHT! and name ' + name + surname)
            else:
                label = 0
                correct_counter = 0
                print('WRONG! and name ' + name + surname)
            # update counters and answers dictionary
            counter += 1
            if correct_counter == 4:
                correct_counter = 0
            # queried.append(itd)
            labels.append(label)
            # first two tests: wrong -> up, right -> same
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
        # print('counter', counter)
        # print('rightmost', rightmost)
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
            print('saving image')
            pc.plot(itds_sorted, labels_sorted, name, surname, unique_itds[seventy_index].item())
            # print(pc.score(itds_sorted, labels_sorted))
            # print(pc.coefs_)
            # print('79.4% point PF curve: ', unique_itds[seventy_index].item())
        return {'wav_location': wavfile, 'itd': itd, 'factor': factor,
            'counter': counter, 'correct_counter': correct_counter, 
            'upsized': upsized, 'downsized': downsized, 'rightmost': rightmost,
            'reversals': reversals, 'downup_reversals': downup_reversals, 
            'queries': queried, 'labels': labels}
    return render_template('test_2afc.html')


'''
if __name__== '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    Session(app)
    app.run(debug=True)

'''
