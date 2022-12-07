document.getElementById("submitPhones").addEventListener("click", function(){
  localStorage.setItem("baldDone", false);
  localStorage.setItem("randomDone", false);
  localStorage.setItem("twoafcDone", false);
  firstname = document.getElementById('firstname').value
  lastname = document.getElementById('lastname').value
  localStorage.setItem("name", firstname);
  localStorage.setItem("lastname", lastname);
  $.ajax({
    url: '/',
    data: {
      'name': firstname,
      'lastname': lastname,
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
    //$('#demo').text(JSON.stringify(data)).show();
  });
});