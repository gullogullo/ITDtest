const totalReverals = 1; //6;

var itd = 0;
var factor = 0;
var counter = 0;
var correctCounter = 0;
var upsized = 0;
var downsized = 0;
var rightmost = 0;
var reversals = 0;
var downupReversals = 0;
var queried = [];
var labels = [];

localStorage.setItem("twoafcDone", "false");

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

function toggleClass() {
  $.ajax({
    url: '/test_2afc',
    data: {
      'answer': 0,
      'itd': itd,
      'factor': factor,
      'counter': counter,
      'correct_counter': correctCounter,
      'upsized': upsized,
      'downsized': downsized,
      'reversals': reversals,
      'downup_reversals': downupReversals,
      'queried_samples': queried,
      'labels': labels,
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
    $('#demo').text(JSON.stringify(data)).show();
    play(data.wav_location);
    itd = data.itd;
    factor = data.factor;
    counter = data.counter;
    correctCounter = data.correct_counter;
    upsized = data.upsized;
    downsized = data.downsized;
    reversals = data.reversals;
    downupReversals = data.downup_reversals;
    queried = data.queries;
    labels = data.labels;
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
  $.ajax({
    url: '/test_2afc',
    data: {
      'answer': 1,
      'itd': itd,
      'factor': factor,
      'counter': counter,
      'correct_counter': correctCounter,
      'upsized': upsized,
      'downsized': downsized,
      'reversals': reversals,
      'downup_reversals': downupReversals,
      'queried_samples': queried,
      'labels': labels,
      'rightmost': rightmost,
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
    if (data.reversals >= totalReverals) {
      localStorage.setItem("twoafcDone", "true");
      redirect('/test_select');
    }
    else {
      setTimeout(function() {document.getElementById("playDown").classList.toggle("goUp"); }, 250);
      itd = data.itd;
      factor = data.factor;
      counter = data.counter;
      correctCounter = data.correct_counter;
      upsized = data.upsized;
      downsized = data.downsized;
      reversals = data.reversals;
      downupReversals = data.downup_reversals;
      queried = data.queries;
      labels = data.labels;
      //$('#demo').text(JSON.stringify(data)).show();
      document.getElementById("btn1").classList.toggle("goUp");
      document.getElementById("btn1").classList.toggle("comeDown");
      document.getElementById("btn2").classList.toggle("goDown");
      document.getElementById("btn2").classList.toggle("comeUp");
    }
  });
}


function toggleBtn2() {
  $.ajax({
    url: '/test_2afc',
    data: {
      'answer': 2,
      'itd': itd,
      'factor': factor,
      'counter': counter,
      'correct_counter': correctCounter,
      'upsized': upsized,
      'downsized': downsized,
      'reversals': reversals,
      'downup_reversals': downupReversals,
      'queried_samples': queried,
      'labels': labels,
      'rightmost': rightmost,
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
    $('#demo').text(JSON.stringify(data)).show();
    if (data.reversals >= totalReverals) {
      localStorage.setItem("twoafcDone", "true");
      redirect('/test_select');
    }
    else {
      setTimeout(function() {document.getElementById("playDown").classList.toggle("goUp"); }, 250);
      itd = data.itd;
      factor = data.factor;
      counter = data.counter;
      correctCounter = data.correct_counter;
      upsized = data.upsized;
      downsized = data.downsized;
      reversals = data.reversals;
      downupReversals = data.downup_reversals;
      queried = data.queries;
      labels = data.labels;
      document.getElementById("btn1").classList.toggle("goUp");
      document.getElementById("btn1").classList.toggle("comeDown");
      document.getElementById("btn2").classList.toggle("goDown");
      document.getElementById("btn2").classList.toggle("comeUp");
    }
  });
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
  var url = file + "?cb=" + new Date().getTime();
  var audio = new Audio(url);   
  audio.play();
}

const playCustom = (() => {
  let context = null;
  return async url => {
    if (context) context.close();
    context = new AudioContext();
    const source = context.createBufferSource();
    source.buffer = await fetch(url)
      .then(res => res.arrayBuffer())
      .then(arrayBuffer => context.decodeAudioData(arrayBuffer));
    $('#demo').text('played!').show();
    source.connect(context.destination);
    source.start();
  };
})();

function toggleExit() {
  document.getElementById("closeB").classList.toggle("goDown", true);
  setTimeout(function() { redirect('/test_select'); }, 500);
}