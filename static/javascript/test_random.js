const al_counter = 2; //25;

var trials = 0;
var Xtrain = [];
var ytrain = [];
var pooldata = [];
var scores = [];
var queries = [];
var labels = [];
var itd = 0;
var rightmost = 0;

localStorage.setItem("randomDone", "false");

var freezeClic = false;

document.addEventListener("click", e => {
  setTimeout(function() {
    freezeClic = true
  }, 10);
  if (freezeClic) {
    e.stopPropagation();
    e.preventDefault();
    //$('#demo').text('clickkkk').show();
  }
  setTimeout(function() {
    freezeClic = false
  }, 1000);
}, true);

var loader = document.getElementById("loader");
loader.style.display = "none";

const button = document.getElementById("playDown");
button.addEventListener("click", toggleClass);
//button.addEventListener("click", setTimeout(function() { 
//  var url = wav_location + "?cb=" + new Date().getTime();
//  playCustom(url);}, 500));

const exit = document.getElementById("closeB");
exit.addEventListener("click", toggleExit);

const button1 = document.getElementById("btn1");
button1.addEventListener("click", toggleBtn1);

const button2 = document.getElementById("btn2");
button2.addEventListener("click", toggleBtn2);

let circularProgress = document.querySelector(".circular-progress");
let progressStartValue = 10;
let progressEndValue = 100;

function toggleClass() {
  $.ajax({
    url: '/test_random',
    data: 
    {
      'answer': 0,
      'trials': trials,
      'poolData_Random': pooldata,
      'queried_samples_Random': queries,
    },
    traditional: true,
    type: 'POST',
    success: function(response){
        //console.log(response);
    },
    error: function(error){
        //console.log(error);
    }
  }).done(function(data) {
    //$('#audioPlayer').attr('src', data.wav_location);
    //$('audio')[0].play();0
    //setTimeout(function() { play(data.wav_location);}, 500);
    var url = data.wav_location + "?cb=" + new Date().getTime();
    playCustom(url);
    trials = data.trials
    itd = data.itd;
    Xtrain = data.Xtrain;
    ytrain =data.ytrain;
    pooldata = data.pooldata;
    scores = data.scores;
    queries = data.queries;
    rightmost = data.rightmost;
  });
  document.getElementById("playDown").classList.remove("comeDown");
  document.getElementById("playDown").classList.toggle("goUp");
  setTimeout(function() { 
    document.getElementById("btn1").classList.remove("goUp"); 
    document.getElementById("btn1").classList.toggle("comeDown");
    document.getElementById("btn2").classList.remove("goDown"); 
    document.getElementById("btn2").classList.toggle("comeUp"); 
  }, 2000);
};

function toggleBtn1() {
  loader.style.display = "block";
  trials += 1;
  $.ajax({
    url: '/test_random',
    data: {
      'answer': 1,
      'trials': trials,
      'rightmost': rightmost,
      'X_train_Random': Xtrain,
      'y_train_Random': ytrain,
      'poolData_Random': pooldata,
      'test_scores_Random': scores,
      'labels_Random': labels,
      'queried_samples_Random': queries,
    },
    traditional: true,
    type: 'POST',
    success: function(response){
        //console.log(response);
    },
    error: function(error){
        //console.log(error);
    }
  }).done(function(data) {
    var progress = (progressStartValue + (trials / al_counter) * (progressEndValue - progressStartValue)) * 3.6;
    //$('#demo').text(JSON.stringify(String(data.reversals).concat(String(progress)))).show();
    circularProgress.style.background = `conic-gradient(transparent ${progress
      }deg, #E74C3C 0deg)`;
    if (trials == al_counter) {
      localStorage.setItem("randomDone", "true");
      redirect('/test_select');
    }
    else {
      setTimeout(function() {
        document.getElementById("playDown").classList.toggle("comeDown");
        document.getElementById("playDown").classList.toggle("goUp"); 
      }, 250);
      trials = data.trials;
      Xtrain = data.Xtrain;
      ytrain =data.ytrain;
      pooldata = data.pooldata;
      scores = data.scores;
      queries = data.queries;
      labels = data.labels;
      document.getElementById("btn1").classList.toggle("goUp");
      document.getElementById("btn1").classList.toggle("comeDown");
      document.getElementById("btn2").classList.toggle("goDown");
      document.getElementById("btn2").classList.toggle("comeUp");
      loader.style.display = "none";
    }
  });
};

function toggleBtn2() {
  loader.style.display = "block";
  trials += 1
  $.ajax({
    url: '/test_random',
    data: {
      'answer': 2,
      'trials': trials,
      'rightmost': rightmost,
      'X_train_Random': Xtrain,
      'y_train_Random': ytrain,
      'poolData_Random': pooldata,
      'test_scores_Random': scores,
      'labels_Random': labels,
      'queried_samples_Random': queries,
    },
    traditional: true,
    type: 'POST',
    success: function(response){
        //console.log(response);
    },
    error: function(error){
        //console.log(error);
    }
  }).done(function(data) {
    var progress = (progressStartValue + (trials / al_counter) * (progressEndValue - progressStartValue)) * 3.6;
    //$('#demo').text(JSON.stringify(String(data.reversals).concat(String(progress)))).show();
    circularProgress.style.background = `conic-gradient(transparent ${progress
      }deg, #E74C3C 0deg)`;
    if (trials == al_counter) {
      localStorage.setItem("randomDone", "true");
      redirect('/test_select');
    }
    else {
      setTimeout(function() {
        document.getElementById("playDown").classList.toggle("comeDown");
        document.getElementById("playDown").classList.toggle("goUp"); 
      }, 250);
      trials = data.trials;
      Xtrain = data.Xtrain;
      ytrain =data.ytrain;
      pooldata = data.pooldata;
      scores = data.scores;
      queries = data.queries;
      labels = data.labels;
      document.getElementById("btn1").classList.toggle("goUp");
      document.getElementById("btn1").classList.toggle("comeDown");
      document.getElementById("btn2").classList.toggle("goDown");
      document.getElementById("btn2").classList.toggle("comeUp");
      loader.style.display = "none";
    }
  });
};

function redirect (url) {
  var ua        = navigator.userAgent.toLowerCase(),
      isIE      = ua.indexOf('msie') !== -1,
      version   = parseInt(ua.substr(4, 2), 10);
  // Internet Explorer 8 and lower
  if (isIE && version < 9) {
      var link = document.createElement('a');
      link.href = url;
      document.body.appendChild(link);
      link.click();
  }
  // All other browsers can use the standard window.location.href (they don't lose HTTP_REFERER like Internet Explorer 8 & lower does)
  else { 
      window.location.href = url; 
  }
};

const playCustom = (() => {
  let context = null;
  return async url => {
    if (context) context.close();
    context = new AudioContext();
    const source = context.createBufferSource();
    source.buffer = await fetch(url)
      .then(res => res.arrayBuffer())
      .then(arrayBuffer => context.decodeAudioData(arrayBuffer));
    //$('#demo').text('played!').show();
    source.connect(context.destination);
    source.start();
  };
})();

function toggleExit() {
  document.getElementById("closeB").classList.toggle("goDown");
  setTimeout(function() { redirect('/test_select'); }, 500);
};