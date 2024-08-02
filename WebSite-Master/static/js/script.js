function validateForm() {
    var username = document.forms["login-form"]["username"].value;
    var password = document.forms["login-form"]["password"].value;
  
    if (username == "") {
      alert("Please enter your username");
      return false;
    }
  
    if (password == "") {
      alert("Please enter your password");
      return false;
    }
  
    return true;
  }
  