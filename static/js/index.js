$(document).ready(function() {

    setTimeout( function(){
                if($('.sample-text').length) {
                    $('.sample-text h2 span').typed({
                        strings: ["", "with GPU-accelerated TensorFlow."],
                        typeSpeed: 50,
                        backSpeed: 20,
                        backDelay: 500,
                        loop: false,
                        contentType: 'html', // or text
                        // defaults to false for infinite loop
                        loopCount: false
                    });
                }

            }, 2500);






    $('select').material_select();



    $('select').change(function() {
        var e = document.getElementById("label_num");
        var num = e.options[e.selectedIndex].value;
        createForm(num);
    });

    $('#form-container').on('click', '#process', function() {
        $('.preloader-wrapper').removeClass('hidden').addClass('active');
        $('.progress').removeClass('hidden');
    });

    $('#run_sample').click(function() {
        $('.undrugged-container, .drugged-container, .microscope').fadeOut();
        $('.gps_ring').delay(500).fadeIn();
        $('.progress').removeClass('hidden');
        setTimeout(function(){
            $('.progress').addClass('hidden');
            $('.gps_ring').hide();
            $('.sample-result').prepend('<figure style="width:100%"><img src="./static/img/sample_result.png" alt="Sample Result" width="100%"><figcaption>The differences of cellular drug response are demonstrated on the scatter plot.</figcaption></figure>');
            $('.sample-result').removeClass('hidden').addClass('animated zoomIn');
        }, 5000)

        $(this).prop("disabled",true);
    });

    $('#upload_btn').click(function() {
        $('html, body').animate({
            scrollTop: $(".custom-upload").offset().top - 80
        }, 600);
    });



function createForm(numOfLabels) {

  if (document.getElementById("form")) {
    var f = document.getElementById("form");
    f.parentNode.removeChild(f);
  }

  if (!document.getElementById("process")) {
    var f = document.createElement("form");
    f.setAttribute('method',"post");
    f.setAttribute('id',"form");
    f.setAttribute('action',"render");
    f.setAttribute('enctype',"multipart/form-data");

    var input_container = document.createElement("div");
        input_container.setAttribute('class',"row");

    for (var i = 0; i < numOfLabels; ++i) {

        var div = document.createElement("div");
        div.setAttribute('class',"input-field col s12");

        var label = document.createElement("label");
        label.setAttribute('for',"label" + parseInt(i + 1));
        label.innerHTML = "Label Name " + parseInt(i + 1);
        var input = document.createElement("input");
        input.setAttribute('type',"text");
        input.setAttribute('name',"label" + parseInt(i + 1));
        input.setAttribute('id',"label" + parseInt(i + 1));
        var input_file = document.createElement("input");
        input_file.setAttribute('type',"file");
        input_file.setAttribute('multiple',"multiple");
        input_file.setAttribute('name',"file" + parseInt(i + 1));
        input_file.setAttribute('accept',"image/*");
        div.appendChild(label);
        div.appendChild(input);
        div.appendChild(input_file);
        input_container.appendChild(div);
    }

        var s = document.createElement("input"); //input element, Submit button
        s.setAttribute('type',"submit");
        s.setAttribute('class',"btn waves-effect waves-light");
        s.setAttribute('id',"process");
        s.setAttribute('value',"Process");
        f.appendChild(input_container);
        f.appendChild(s);



        document.getElementById('form-container').appendChild(f);
    }


}


 });


//    document.getElementById("process").addEventListener("click", function(e) {
//    if( document.getElementById("file").files.length == 0 ){
//
//        e.preventDefault();
//        alert("Please select files!");
//    }




