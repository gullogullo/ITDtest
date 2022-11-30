var freezeClic = false;

document.addEventListener("click", e => {
  setTimeout(function() {
    freezeClic = true
  }, 10);
  if (freezeClic) {
    e.stopPropagation();
    e.preventDefault();
    $('#demo').text('clickkkk').show();
  }
  setTimeout(function() {
    freezeClic = false
  }, 1000);
}, true);


const button = document.getElementById("playDown");
button.addEventListener("click", toggleClass);

const exit = document.getElementById("closeB");
exit.addEventListener("click", toggleExit);

const button1 = document.getElementById("btn1");
button1.addEventListener("click", toggleBtn1);

const button2 = document.getElementById("btn2");
button2.addEventListener("click", toggleBtn2);

localStorage.setItem("baldDone", "false");
const al_counter = 4;

var trials = 0;
var Xtrain = [];
var ytrain = [];
var pooldata = [];
var scores = [];
var queries = [];
var labels = [];
var itd = 0;
var rightmost = 0;

function toggleClass() {
  $.ajax({
    url: '/test_bald',
    data: 
    {
      'answer': 0,
      'trials': trials,
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
    trials = data.trials
    itd = data.itd;
    Xtrain = data.Xtrain;
    ytrain =data.ytrain;
    pooldata = data.pooldata;
    scores = data.scores;
    queries = data.queries;
    rightmost = data.rightmost;
  });

  document.getElementById("playDown").classList.toggle("goUp");
  setTimeout(function() {
    document.getElementById("btn1").classList.remove("goUp"); 
    document.getElementById("btn1").classList.toggle("comeDown");
    document.getElementById("btn2").classList.remove("goDown"); 
    document.getElementById("btn2").classList.toggle("comeUp"); 
  }, 2000);
}


function toggleBtn1() {
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
    //$('#demo').text(JSON.stringify(labels)).show();
    if (trials == al_counter) {
      localStorage.setItem("baldDone", "true");
      redirect('/test_select');
    }
    else {
      setTimeout(function() {document.getElementById("playDown").classList.toggle("goUp"); }, 250);
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
    }
  });
}


function toggleBtn2() {
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
    //$('#demo').text(JSON.stringify(labels)).show();
    if (trials == al_counter) {
      localStorage.setItem("baldDone", "true");
      redirect('/test_select');
    }
    else {
      setTimeout(function() {document.getElementById("playDown").classList.toggle("goUp"); }, 250);
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
    }
  });
}


function toggleExit() {
  document.getElementById("closeB").classList.toggle("goDown", true);
  setTimeout(function() { redirect('/test_select'); }, 500);
}


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
}

function play(file) {
  var audio = new Audio(file);
  audio.play();
}