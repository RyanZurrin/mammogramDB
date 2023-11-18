'use strict';

cornerstoneWADOImageLoader.external.cornerstone = cornerstone;
cornerstoneWADOImageLoader.external.dicomParser = dicomParser;
cornerstoneTools.external.Hammer = Hammer;
cornerstoneTools.external.cornerstone = cornerstone;
cornerstoneTools.external.cornerstoneMath = cornerstoneMath;

let score = 0;
let quality = "Good"
let id;

window.oncontextmenu = function (e) {
    e.preventDefault();
    return false;
}

window.onload = function () {
    let x = new XMLHttpRequest();
    x.open("GET", "/get_next_dicom", true);
    x.send();
    x.onload = function (e) {
        let res = e.currentTarget.responseText;
        res = JSON.parse(res);
        id = res.msg;
        displayDicom(id);
    }
}

function updateCoords(e) {
    if (checkToolState()) {
        const crds = getCoords();
        let coords = ("(" + crds[0] + "," + crds[1] + "), (" + crds[2] + "," + crds[3] + ")")
        updateImageData('coords', coords);
    } else {
        updateImageData('coords', "")
    }
}

// listen for the left mouse button up event
document.addEventListener('mouseup', function (e) {
    // if the left mouse up event is triggered update the coords on screen
    if (e.button === 0) {
        updateCoords(e);
    }
}, false);

//listen for the left mouse button down event
document.addEventListener('mousedown', function (e) {
    document.addEventListener('mousemove', function (e) {
        if (e.button === 0) {
            updateCoords(e);
        }
    }, false);
}, false);


window.onkeydown = function (e) {
    e.preventDefault();
    processKeyPress(e.code);
};

function checkToolState() {
    let toolState = getToolState();
    if (toolState === undefined) {
        return false;
    } else {
        return true;
    }
}

function getCoords() {
    if (checkToolState()) {
        let toolState = getToolState();
        let boxStartX = Math.floor(toolState.RectangleRoi.data[0].handles.start.x)
        let boxStartY = Math.floor(toolState.RectangleRoi.data[0].handles.start.y)
        let boxEndX = Math.floor(toolState.RectangleRoi.data[0].handles.end.x)
        let boxEndY = Math.floor(toolState.RectangleRoi.data[0].handles.end.y)
        return [boxStartX, boxStartY, boxEndX, boxEndY]
    } else {
        return [0, 0, 0, 0]
    }
}

function displayDicom(id) {
    let viewer = document.getElementById('viewer');
    let imageId =
        "dicomweb://" + window.location.host + "/get_dicom/?id=" + id;
    cornerstone.enable(viewer);
    cornerstone.loadImage(imageId).then(function (image) {
        console.log('Loaded', image);
        cornerstoneTools.init();
        let viewer = document.getElementById('viewer');
        cornerstone.enable(viewer);
        cornerstone.displayImage(viewer, image);
        cornerstoneTools.addTool(cornerstoneTools.RectangleRoiTool);
        cornerstoneTools.setToolActive('RectangleRoi', {mouseButtonMask: 1});
        cornerstoneTools.addTool(cornerstoneTools.ZoomTool);
        cornerstoneTools.setToolActive('Zoom', {mouseButtonMask: 2});
        cornerstoneTools.addTool(cornerstoneTools.PanTool);
        cornerstoneTools.setToolActive('Pan', {mouseButtonMask: 4});
        cornerstoneTools.addTool(cornerstoneTools.WwwcTool);
    });
    updateImageData("sop_id", id);
}

function updateImageData(id, value) {
    let tooltip = document.getElementById(id);
    tooltip.innerHTML = value;
}

function clearROI() {
    let toolstate;
    toolstate = {}
    cornerstoneTools
        .globalImageIdSpecificToolStateManager
        .restoreToolState(toolstate);
    const elem = document.getElementById('viewer');
    cornerstone.updateImage(elem);
}

function setURL(x1, y1, x2, y2, id, score, quality) {
    return window.location.protocol + "//" + window.location.host +
        "/roi?x1=" + x1 + "&y1=" + y1 + "&x2=" +
        x2 + "&y2=" + y2 + "&id=" + id + "&score=" +
        score + "&quality=" + quality;
}

function changeRoiColor(color) {
    let toolState = getToolState();
    toolState.RectangleRoi.data[0].color = color;
    const elem = document.getElementById('viewer');
    cornerstone.updateImage(elem);
}

function getToolState() {
    let toolState = cornerstoneTools.globalImageIdSpecificToolStateManager.saveToolState();
    let keys = Object.keys(toolState);
    return toolState[keys[0]];
}

function resetInputData(quality, score, coords) {
    clearROI();
    updateImageData("score", score);
    updateImageData("quality", quality);
    updateImageData('coords', coords);
}

function processKeyPress(e) {
    switch (e) {
        case "Digit0":
            score = .00;
            changeRoiColor("#fff");
            updateImageData("score", ".00")
            break;
        case "Digit1":
            score = .20;
            changeRoiColor("#fd0b0b");
            updateImageData("score", ".20")
            break;
        case "Digit2":
            score = .40;
            changeRoiColor("#f99209");
            updateImageData("score", ".40")
            break;
        case "Digit3":
            score = .60;
            changeRoiColor("#ecf60c");
            updateImageData("score", ".60")
            break;
        case "Digit4":
            score = .80;
            changeRoiColor("#2efb0d");
            updateImageData("score", ".80")
            break;
        case "Digit5":
            score = 1.00;
            changeRoiColor("#2828f1");
            updateImageData("score", "1.00")
            break;
        case "KeyX":
            quality = "Good"
            resetInputData(quality, 0.0, "");
            break;
        case "KeyZ":
            if (quality === "Good") {
                quality = "Bad";
                resetInputData(quality, 0.0, "")
            } else {
                quality = "Good";
                updateImageData("quality", "Good")
            }
            break;
        case "KeyR":
            displayDicom(id);
            break;
        case "ControlLeft":
        case "ControlRight":
            document.addEventListener('keydown', function (e) {
                if (e.ctrlKey) {
                    cornerstoneTools.setToolActive('Wwwc', {mouseButtonMask: 1});
                }
            }, false);
            document.addEventListener('keyup', function (e) {
                if (!e.ctrlKey) {
                    cornerstoneTools.setToolActive('RectangleRoi', {mouseButtonMask: 1});
                }
            }, false);
            break;
        case "Space":
        case "Enter":
            let url;
            const crds = getCoords();
            url = setURL(crds[0], crds[1], crds[2], crds[3], id, score, quality);
            let x = new XMLHttpRequest();
            x.open("GET", url, true);
            x.send();
            x.onload = function (e) {
                window.location.href = "/";
            }
            break;
        default:
            break;
    }
}