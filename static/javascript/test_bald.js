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

localStorage.setItem("baldDone", "false");

var freezeClic = false;

document.addEventListener("click", e => {
  setTimeout(function() {
    freezeClic = true
  }, 10);
  if (freezeClic) {
    e.stopPropagation();
    e.preventDefault();
    //('#demo').text('clickkkk').show();
  }
  setTimeout(function() {
    freezeClic = false
  }, 1000);
}, true);

var loader = document.getElementById("loader");
loader.style.display = "none";

const button = document.getElementById("playDown");
button.addEventListener("click", toggleClass);

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
    url: '/test_bald',
    data: 
    {
      'answer': 0,
      'trials': trials,
      'poolData_Bald': pooldata,
      'queried_samples_Bald': queries,
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
    //$('audio')[0].play();
    //setTimeout(function() { play(data.wav_location);}, 500);
    //play(data.wav_location);
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
    url: '/test_bald',
    data: {
      'answer': 1,
      'trials': trials,
      'rightmost': rightmost,
      'X_train_Bald': Xtrain,
      'y_train_Bald': ytrain,
      'poolData_Bald': pooldata,
      'test_scores_Bald': scores,
      'labels_Bald': labels,
      'queried_samples_Bald': queries,
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
      localStorage.setItem("baldDone", "true");
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
  trials += 1;
  $.ajax({
    url: '/test_bald',
    data: {
      'answer': 2,
      'trials': trials,
      'rightmost': rightmost,
      'X_train_Bald': Xtrain,
      'y_train_Bald': ytrain,
      'poolData_Bald': pooldata,
      'test_scores_Bald': scores,
      'labels_Bald': labels,
      'queried_samples_Bald': queries,
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
      localStorage.setItem("baldDone", "true");
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

function play(file) {
  var url = file + "?cb=" + new Date().getTime();
  var audio = new Audio(url);
  audio.play();
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
}