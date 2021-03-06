$(document).ready(function(){
  $('#myModal').modal({ show: false});
  // setTimeout(function(){
  //   $(".filedrag").on("drop",function(event){
  //     event.preventDefault();
     
  //     alert("dropoped");
  //   });
  // },2000);
});
$(function(){
  var viewModel = {};
  viewModel.fileData = ko.observable({
    dataURL: ko.observable(),
    // base64String: ko.observable(),
  });
  viewModel.multiFileData = ko.observable({
    dataURLArray: ko.observableArray(),
  });
  viewModel.onClear = function(fileData){
    if(confirm('Are you sure?')){
      fileData.clear && fileData.clear();
    }                            
  };
  viewModel.debug = function(){
    window.viewModel = viewModel;
    console.log(ko.toJSON(viewModel));
    debugger; 
  };
  ko.applyBindings(viewModel);
});

// $('#uploadForm').submit(function (e) {
// 	console.log("Inside Upload Form");
// 	var formData = new FormData();
// 	var img = $('#file')[0].files[0];
	
// 	formData.append('file', $('#file')[0].files[0]);
//   if(img){
//     $('#myModal').modal('show');
//     $.ajax({
//       url : 'https://deloitte-image-analytics.herokuapp.com/htmlimage',
//       type : 'POST',
//       data : formData,	
//       processData: false,  // tell jQuery not to process the data
//       contentType: false,  // tell jQuery not to set contentType
//          success : function(data) {
//           console.log(data);
//           createTable(data);
//           showImage(data, img);
//           $('#myModal').modal('hide');
//          },
//          error: function(request,status,errorThrown) {
//            alert("inside error");
//           $('#myModal').modal('hide');    
//          }
//     });
//   }
//   else{
//    e.preventDefault();
//   }
// 	return false;
// });

$('#uploadForm').submit(function () {
	console.log("Inside Upload Form");
	var formData = new FormData();
	var img = $('#file')[0].files[0];
	console.log(img);
	formData.append('file', $('#file')[0].files[0]);
// formData.append('file2', $('#file2')[0].files[0]);
  console.log("Inside Submit");
 
  
    $('#myModal').modal('show');
	$.ajax({
		   url : 'https://deloitte-image-analytics.herokuapp.com/htmlimage',
		   type : 'POST',
		   data : formData,	
		   processData: false,  // tell jQuery not to process the data
		   contentType: false,  // tell jQuery not to set contentType
		   success : function(data) {
				console.log(data);
				createTable(data);
        showImage(data, img);
        $('#myModal').modal('hide');   
		   }
  });
  
	return false;
});

$('#uploadSimilarityForm').submit(function () {
	console.log("Inside Upload Form");
	var formData = new FormData();
	var img = $('#file')[0].files[0];
	console.log(img);
	formData.append('file', $('#file')[0].files[0]);
    formData.append('file2', $('#file2')[0].files[0]);
  console.log("Inside Submit");


    $('#myModal').modal('show');
	$.ajax({
		   url : 'https://deloitte-image-analytics.herokuapp.com/similarimage',
		   type : 'POST',
		   data : formData,
		   processData: false,  // tell jQuery not to process the data
		   contentType: false,  // tell jQuery not to set contentType
		   success : function(data) {
				console.log(data);
				createTable(data);
        showImage(data, img);
        $('#myModal').modal('hide');
		   }
  });

	return false;
});

function createTable(data){
	var container = $('#showTable'),
	table = $('<table class="table table-hover table-striped">');
	thead=$('<thead>').append('<th>Category</th><th>Probability</th>');
	tbody=$('<tbody>');
	console.log("Inside" + jQuery.parseJSON(data['results'][0]).category);
	for (i = 0; i < data['results'].length; i++) {
	var jsonData = jQuery.parseJSON(data['results'][i]);
	var tr = $('<tr>');
    tr.append('<td>' + jsonData.category + '</td>').append('<td>' + jsonData.probability + '</td>');
	tbody.append(tr);
}
table.append(thead);
table.append(tbody);
container.empty();
container.append(table);
}


function showSimilarity(data){
	var container = $('#showTable'),
	table = $('<table class="table table-hover table-striped">');
	thead=$('<thead>').append('<th>Category</th><th>Probability</th>');
	tbody=$('<tbody>');
	console.log("Inside" + jQuery.parseJSON(data['results'][0]).category);
	for (i = 0; i < data['results'].length; i++) {
	var jsonData = jQuery.parseJSON(data['results'][i]);
	var tr = $('<tr>');
    tr.append('<td>' + jsonData.category + '</td>').append('<td>' + jsonData.probability + '</td>');
	tbody.append(tr);
}
table.append(thead);
table.append(tbody);
container.empty();
container.append(table);
}


function showImage(data, imgFile){
var ctx = document.getElementById('myCanvas').getContext('2d');
    var url = URL.createObjectURL(imgFile);
    var img = new Image();
    img.onload = function() {
        ctx.drawImage(img,10,10,300,300);
        var canvas = document.getElementById('myCanvas');
        var context = canvas.getContext('2d');
        for (i = 0; i < data['results'].length; i++) {
            var jsonData = jQuery.parseJSON(data['results'][i]);
            var width = jsonData.xmax - jsonData.xmin;
            var height = jsonData.ymax - jsonData.ymin;
            context.beginPath();
            context.rect(parseInt(jsonData.xmin)+10, parseInt(jsonData.ymin)+10, width, height);
            context.lineWidth="2";
            context.strokeStyle = 'yellow';
            context.stroke();
        }
    }
    img.src = url;


}