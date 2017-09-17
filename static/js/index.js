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

$('#uploadForm').submit(function () {
	console.log("Inside Upload Form");
	var formData = new FormData();
	var img = $('#file')[0].files[0];
	console.log(img);
	formData.append('file', $('#file')[0].files[0]);
	console.log("Inside Submit");
	$.ajax({
		   url : 'http://localhost:5000/htmlimage',
		   type : 'POST',
		   data : formData,	
		   processData: false,  // tell jQuery not to process the data
		   contentType: false,  // tell jQuery not to set contentType
		   success : function(data) {
				console.log(data);
				createTable(data);
				showImage(data, img);
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