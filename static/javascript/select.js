const exit = document.getElementById("closeB");
exit.addEventListener("click", toggleExit);

bald = localStorage.getItem("baldDone");
random= localStorage.getItem("randomDone");
twoafc = localStorage.getItem("twoafcDone");

username = localStorage.getItem("name");
usersurname = localStorage.getItem("lastname");

/*
const urlstatic = "static/figures/";
var url1 = urlstatic.concat(String(username));
var url2 = url1.concat("_".concat(String(usersurname)));
const baldString = "_PF_BALD_Approximation.png";
const randomString = "_PF_Random_Approximation.png";
const twoafcString = "_PF_WH_Approximation.png";
var urls = [url2.concat(baldString), url2.concat(randomString), url2.concat(twoafcString)];
*/

const urlcsv = "static/csvs/";
var urlcsv1 = urlcsv.concat(String(username));
var urlcsv2 = urlcsv1.concat("_".concat(String(usersurname)));
//const baldCsvString = "_2afc_results.csv";
//const randomCsvString = "_bald_results.csv";
//const twoafcCsvString = "_random_results.csv";
//var urlsCsv = [urlcsv2.concat(baldCsvString), urlcsv2.concat(randomCsvString), urlcsv2.concat(twoafcCsvString)];
var urlsCsv = [urlcsv2.concat("_results.csv")]

if (bald == "true") {
  //document.getElementById("baldPlot").setAttribute("src", url2.concat(baldString)); 
  document.getElementById("baldPlot").style.display = "block";
  document.getElementById("bald").style.display = "none";
}
else {
  document.getElementById("baldPlot").style.display = "none";
};

if (random == "true") {
  //document.getElementById("randomPlot").setAttribute("src", url2.concat(randomString));
  document.getElementById("randomPlot").style.display = "block";
  document.getElementById("random").style.display = "none";
}
else {
  document.getElementById("randomPlot").style.display = "none";
};

if (twoafc == "true") {
  //document.getElementById("twoafcPlot").setAttribute("src", url2.concat(twoafcString)); 
  document.getElementById("twoafcPlot").style.display = "block";
  document.getElementById("twoafc").style.display = "none";
}
else {
  document.getElementById("twoafcPlot").style.display = "none";
};

if (bald == "true" && random == "true" && twoafc == "true") {
  document.getElementById("twoafc").style.display = "none";
  document.getElementById("random").style.display = "none";
  document.getElementById("bald").style.display = "none";
  if (confirm('Do you want to save the results?')) {
    var interval = setInterval(download, 300, urlsCsv);
  } else {
    // Do nothing!
    //console.log('Plot not saved');
    setTimeout(function() { redirect('/'); }, 3000);
  };
  setTimeout(function() { redirect('/'); }, 8000);
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
    // setTimeout(function() { redirect('/'); }, 1000);
  }
};

function toggleExit() {
  document.getElementById("closeB").classList.toggle("goDown");
  localStorage.setItem("baldDone", false);
  localStorage.setItem("randomDone", false);
  localStorage.setItem("twoafcDone", false);
  setTimeout(function() { redirect('/'); }, 500);
};