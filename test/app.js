var canvas;

$(function () {

    var model;

    async function prepareModel() {
        console.log("Download model...");
		model = await tf.loadModel('https://raw.githubusercontent.com/christian-vorhemus/shape-classification/master/model/models/shapedetection_model/model.json');
        console.log("Model downloaded");
    }
    
	async function predict(svg) {
        
        console.log("Start prediction...")
        var sequence = getCoordinates(svg);
        var input = tf.tensor3d([sequence]);
        var pred = await model.predict(input);

        var predCircle = pred.dataSync()[0];
        var predLine = pred.dataSync()[1];
        var predRectangle = pred.dataSync()[2];

        console.log("Circle score: " + predCircle);
        console.log("Line score: " + predLine);
        console.log("Rectangle score: " + predRectangle);

        if(predCircle > 0.85) {
            console.log("Circle");
            createCircle(svg);
            removeObjects();
        } else if(predLine > 0.9) {
            console.log("Line");
            createLine(svg);
            removeObjects();
        } else if(predRectangle > 0.8) {
            console.log("Rectangle");
            createRectangle(svg);
            removeObjects();
        } else {
            console.log("No match");
            freezeObjects();
        }
	}

    function removeObjects() {
        canvas.getObjects().forEach(function (obj) {
            if(obj.excludeFromExport == false || obj.excludeFromExport == undefined) {
                canvas.remove(obj);
            }
            obj.set("excludeFromExport", true);
        });
    }

    function freezeObjects() {
        canvas.getObjects().forEach(function (obj) {
            obj.set("excludeFromExport", true);
        });
    }

    // Return startcoordinates, endcoordinates, min-x, max-x, min-y, max-y
    function getKeyCoordinates(svg) {

        var parser = new DOMParser();
        var doc = parser.parseFromString(svg, "image/svg+xml");

        var paths = doc.getElementsByTagName("path");
        sequence = [];

        var startCoordinates = [];
        var endCoordinates = [];
        var currMinX = null;
        var currMinY = null;
        var currMaxX = null;
        var currMaxY = null;

        $.each(paths, function(index, element) {
            var path = element.getAttribute("d");

            var first = path.indexOf("Q");
            var last = path.lastIndexOf("L");
        
            var subpaths = path.substring(first, last).split("Q ");
            subpaths.shift();

            $.each(subpaths, function(index, subpath) {
                var coordinates = subpath.split(" ");

                var x = parseFloat(coordinates[0]);
                var y = parseFloat(coordinates[1]);

                if(currMinX == null || x < currMinX) {
                    currMinX = x;
                }
                if(currMinY == null || y < currMinY) {
                    currMinY = y;
                }
                if(currMaxX == null || x > currMaxX) {
                    currMaxX = x;
                }
                if(currMaxY == null || y > currMaxY) {
                    currMaxY = y;
                }

                if(index == 0) {
                    startCoordinates = [x,y];
                }

                if(subpaths.length-1 == index) {
                    endCoordinates = [x,y];
                }

            });
        });

        return [startCoordinates, endCoordinates, [currMinX, currMinY], [currMaxX, currMaxY]];
    }

    function getCoordinates(svg) {
        var parser = new DOMParser();
        var doc = parser.parseFromString(svg, "image/svg+xml");

        var paths = doc.getElementsByTagName("path");
        sequence = []

        $.each(paths, function(index, element) {
            var path = element.getAttribute("d");

            var first = path.indexOf("Q");
            var last = path.lastIndexOf("L");
        
            var subpaths = path.substring(first, last).split("Q ");
            subpaths.shift();
        
            var xPrevious = 0
            var yPrevious = 0
            var start = true;
            var i = 0

            $.each(subpaths, function(index, subpath) {
                var coordinates = subpath.split(" ")
                try {
                    var x = coordinates[0];
                    var y = coordinates[1];
        
                    if(start) {
                        xPrevious = x;
                        yPrevious = y;
                        start = false
                    }
        
                    var xOld = x;
                    var yOld = y;
                    x = parseFloat(x) - parseFloat(xPrevious);
                    y = parseFloat(y) - parseFloat(yPrevious);
                    xPrevious = xOld;
                    yPrevious = yOld;
        
                    // Only add every second point to make object simpler
                    if(i%2 == 0) {
                        sequence.push([x,y]);
                    }
                    i++;
                } catch(e) {
                    console.log(e);
                }
            });
    
        });

        var maxSequenceLength = 397;

        if(sequence.length >= maxSequenceLength) {
            sequence = sequence.slice(0, maxSequenceLength);
        } else {
            for(var i=sequence.length;i<maxSequenceLength;i++) {
                sequence.push([0,0]);
            }
        }

        return sequence;
    }

    canvas = window._canvas = new fabric.Canvas('canvas');
    canvas.backgroundColor = '#efefef';
    canvas.isDrawingMode = 1;
    canvas.freeDrawingBrush.color = "black";
    canvas.freeDrawingBrush.width = 5;
    canvas.renderAll();

    var predictionInterval = null;

    function makePrediction() {
        clearInterval(predictionInterval);
        var svg = canvas.toSVG();
        predict(svg);
    }

    canvas.on('mouse:up', function(options) {
        if(predictionInterval == null) {
            predictionInterval = setInterval(makePrediction, 1000);
        } else {
            clearInterval(predictionInterval);
            predictionInterval = setInterval(makePrediction, 1000);
        }
    });

    document.getElementById('clear').addEventListener('click', function (e) {
        canvas.clear();
        canvas.backgroundColor = '#efefef';
    });

    function createCircle(svg) {

        var coordinates = getKeyCoordinates(svg);

        var left = coordinates[2][0];
        var top = coordinates[2][1];
        var radius = (coordinates[3][0] - coordinates[2][0])/2;

        canvas.add(new fabric.Circle({
            left: left,
            top: top,                
            radius: radius,
            stroke: 'black',
            strokeWidth: 5,
            fill: '',
            excludeFromExport: true
        }));

    };

    function createLine(svg) {

        var coordinates = getKeyCoordinates(svg);

        var startX = coordinates[0][0];
        var startY = coordinates[0][1];
        var endX = coordinates[1][0];
        var endY = coordinates[1][1];

        canvas.add(new fabric.Line([startX, startY, endX, endY], {
            stroke: 'black',
            strokeWidth: 5,
            excludeFromExport: true
        }));

    };

    function createRectangle(svg) {

        var coordinates = getKeyCoordinates(svg);

        var left = coordinates[2][0];
        var top = coordinates[2][1];
        var width = (coordinates[3][0] - coordinates[2][0]);
        var height = (coordinates[3][1] - coordinates[2][1]);

        canvas.add(new fabric.Rect({
            left: left,
            top: top,
            width: width,
            height: height,
            fill: '',
            stroke: 'black',
            strokeWidth: 5,
            excludeFromExport: true
         }));

    };

    prepareModel();

});