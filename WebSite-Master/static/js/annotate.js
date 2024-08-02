// Define variables for canvas and fabric.js objects
var canvas = new fabric.Canvas('canvas');
var image = document.getElementById('image');
var canvasWidth = canvas.getWidth();
var canvasHeight = canvas.getHeight();

// Set canvas dimensions to match image dimensions
canvas.setWidth(image.width);
canvas.setHeight(image.height);

// Load image onto canvas
fabric.Image.fromURL(image.src, function(img) {
    canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas), {
        scaleX: canvas.width / img.width,
        scaleY: canvas.height / img.height
    });
});

// Define variables for annotation rectangle
var rect;
var isDown = false;
var origX, origY;

// Function to create annotation rectangle on mouse down event
function onMouseDown(event) {
    origX = event.e.clientX - canvas._offset.left;
    origY = event.e.clientY - canvas._offset.top;
    isDown = true;
    rect = new fabric.Rect({
        left: origX,
        top: origY,
        width: 0,
        height: 0,
        fill: 'rgba(255, 255, 255, 0)',
        stroke: 'red',
        strokeWidth: 2
    });
    canvas.add(rect);
}

// Function to update annotation rectangle on mouse move event
function onMouseMove(event) {
    if (!isDown) return;
    var mouseX = event.e.clientX - canvas._offset.left;
    var mouseY = event.e.clientY - canvas._offset.top;
    var width = mouseX - origX;
    var height = mouseY - origY;
    if (width > 0 && height > 0) {
        rect.set({ width: width, height: height });
    }
    else if (width > 0 && height < 0) {
        rect.set({ left: origX, top: mouseY, width: width, height: -height });
    }
    else if (width < 0 && height > 0) {
        rect.set({ left: mouseX, top: origY, width: -width, height: height });
    }
    else {
        rect.set({ left: mouseX, top: mouseY, width: -width, height: -height });
    }
    canvas.renderAll();
}

// Function to complete annotation rectangle on mouse up event
function onMouseUp(event) {
    isDown = false;
    rect.setCoords();
}

// Add event listeners for mouse down, move, and up events on canvas
canvas.on('mouse:down', onMouseDown);
canvas.on('mouse:move', onMouseMove);
canvas.on('mouse:up', onMouseUp);

// Function to save annotation and send back to Flask app
function saveAnnotation() {
    var annotation = {
        x: rect.left / image.width,
        y: rect.top / image.height,
        width: rect.width / image.width,
        height: rect.height / image.height
    };
    var xhr = new XMLHttpRequest();
    xhr.open('POST', '{{ url_for("annotate") }}');
    xhr.setRequestHeader('Content-Type', 'application/json;charset=UTF-8');
    xhr.send(JSON.stringify(annotation));
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4 && xhr.status === 200) {
            window.location.href = '{{ url_for("index") }}';
        }
    }
}

// Add event listener for save button
var saveButton = document.getElementById('save-button');
saveButton.addEventListener('click', saveAnnotation);

// Add event listener for cancel button
var cancelButton = document.getElementById('cancel-button');
cancelButton.addEventListener('click', function() {
    window.location.href = '{{ url_for("index") }}';
});

// Function to handle window resize events and update canvas dimensions
function onWindowResize(event) {
    canvas.setWidth(image.width);
    canvas.setHeight(image.height);
    canvas.setBackgroundImage(image.src, canvas.renderAll.bind(canvas), {
        scaleX: canvas.width / image.width,
        scaleY: canvas.height / image.height
    });
    rect.setCoords();
}

// Add event listener for window resize events
window.addEventListener('resize', onWindowResize);

// Function to handle canvas zoom in and zoom out events
function onCanvasZoom(event) {
    var delta = event.deltaY;
    var zoom = canvas.getZoom();
    zoom *= 0.999 ** delta;
    if (zoom > 20) zoom = 20;
    if (zoom < 0.01) zoom = 0.01;
    canvas.setZoom(zoom);
    event.preventDefault();
    event.stopPropagation();
}

// Add event listener for canvas zoom events
canvas.on('mouse:wheel', onCanvasZoom);

// Function to handle canvas panning events
function onCanvasPan(event) {
    if (event.e.type === 'mousedown') {
        var posX = event.e.clientX;
        var posY = event.e.clientY;
        canvas.on('mouse:move', function(event) {
            canvas.setCursor('grabbing');
            canvas.viewportTransform[4] += event.e.clientX - posX;
            canvas.viewportTransform[5] += event.e.clientY - posY;
            posX = event.e.clientX;
            posY = event.e.clientY;
            canvas.renderAll();
        });
    }
    else {
        canvas.setCursor('default');
        canvas.off('mouse:move');
    }
}

// Add event listener for canvas pan events
canvas.on('mouse:down', onCanvasPan);
canvas.on('mouse:up', onCanvasPan);
