
document.addEventListener("DOMContentLoaded", function() {

    document.getElementById("process").addEventListener("click", function(e) {
    if( document.getElementById("file").files.length == 0 ){

        e.preventDefault();
        alert("Please select a file!");
    }

    })


});