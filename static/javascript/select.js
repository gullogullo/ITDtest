const exit = document.getElementById("closeB");
exit.addEventListener("click", toggleExit);

bald = localStorage.getItem("baldDone");
random= localStorage.getItem("randomDone");
twoafc = localStorage.getItem("twoafcDone");

username = localStorage.getItem("name");
usersurname = localStorage.getItem("lastname");

//const urlstatic = "{{ url_for('static', filename='figures/";
const urlstatic = "static/figures/";
var url1 = urlstatic.concat(String(username));
var url2 = url1.concat("_".concat(String(usersurname)));
const baldString = "_PF_BALD_Approximation.png";
const randomString = "_PF_Random_Approximation.png";
const twoafcString = "_PF_WH_Approximation.png";

var urls = [url2.concat(baldString), url2.concat(randomString), url2.concat(twoafcString)];

//$('#demo').text(url2.concat(baldString)).show();
//document.getElementById("demo").innerHTML = username;

if (bald == "true") {
  document.getElementById("baldPlot").setAttribute("src", url2.concat(baldString)); 
  document.getElementById("bald").style.display = "none";
}
else {
  document.getElementById("baldPlot").style.display = "none";
};

if (random == "true") {
  document.getElementById("randomPlot").setAttribute("src", url2.concat(randomString)); 
  document.getElementById("random").style.display = "none";
}
else {
  document.getElementById("randomPlot").style.display = "none";
};

if (twoafc == "true") {
  document.getElementById("twoafcPlot").setAttribute("src", url2.concat(twoafcString)); 
  document.getElementById("twoafc").style.display = "none";
}
else {
  document.getElementById("twoafcPlot").style.display = "none";
};

if (bald == "true" && random == "true" && twoafc == "true") {
  if (confirm('Do you want to save the plot?')) {
    var interval = setInterval(download, 300, urls);
    setTimeout(function() { redirect('/'); }, 5000);
    //console.log('Plot saved');
  } else {
    // Do nothing!
    //console.log('Plot not saved');
  }
    setTimeout(function() { redirect('/'); }, 300);
};

function toggleExit() {
  document.getElementById("closeB").classList.toggle("goDown", true);
  localStorage.setItem("baldDone", false);
  localStorage.setItem("randomDone", false);
  localStorage.setItem("twoafcDone", false);
  setTimeout(function() { redirect('/'); }, 500);
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

function download(urls) {
  var url = urls.pop();
  var a = document.createElement('a');
  a.href = url;
  a.download = url.split('/').pop();
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  if (urls.length == 0) {
    clearInterval(interval);
  }
};