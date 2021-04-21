import React, { useState } from 'react';

// https://github.com/Gherciu/react-p5
import Sketch from 'react-p5';
import { mapRange } from '../../utils';

// access tensorflow functions
import * as tf from '@tensorflow/tfjs';

export default function SketchWrapper() {
  const [canvWidth, setCanvWidth] = useState(500);
  const [canvHeight, setCanvHeight] = useState(500);

  // y = a + bx + cx + dx^2
  const [aFeature, setAfeature] = useState(0);
  const [bFeature, setBfeature] = useState(0);
  const [cFeature, setCfeature] = useState(0);
  const [dFeature, setDfeature] = useState(0);

  // _underscore vals is not a tensor
  const [x_vals, setX_vals] = useState([]);
  const [y_vals, setY_vals] = useState([]);

  const [dragging, setDragging] = useState(false);

  // cartesian coordinates
  const initCoordinate = -1;
  const endCoordinate = 1;

  const learningRate = 0.1;

  // optimiser: (sgd) stochastic gradient descent
  // function will implement an algorithm
  // that will adjust our coefficient values
  // based on the output of the loss function.
  const optimizer = tf.train.sgd(learningRate);
  // const optimizer = tf.train.adam(learningRate);

  // loss function: mean squared error
  // the loss function will measure how well
  // our linear equation fits the data.
  // A lower loss value = closer fit.
  const loss = (yPred, yLabels) => {
    return yPred.sub(yLabels).square().mean();
  };

  const predict = (x) => {
    tf.tidy(() => {
      // Create a vector of x values
      const tensor_xVector = tf.tensor1d(x);

      // y = ax^3 + bx^2 + cx + d
      const tensor_yPred = tensor_xVector
        .pow(tf.scalar(3))
        .mul(aFeature)
        .add(tensor_xVector.square().mul(bFeature))
        .add(tensor_xVector.mul(cFeature))
        .add(dFeature);

      // gives back a tensor
      return tensor_yPred;
    });
  };

  // handle click
  function mousePressed(mouseX, mouseY) {
    // map normalize clicked pixels between -1 to 1
    let new_mouseX = mapRange(
      mouseX,
      0,
      canvWidth,
      initCoordinate,
      endCoordinate
    );
    let new_mouseY = mapRange(
      mouseY,
      0,
      canvHeight,
      endCoordinate,
      initCoordinate
    );

    setX_vals((prevArr) => [...prevArr, new_mouseX]);
    setY_vals((prevArr) => [...prevArr, new_mouseY]);
  }

  function onDragged(data) {
    console.log(data);
  }

  // function mousePressed() {
  //   setDragging(true);
  // }
  // function mouseReleased() {
  //   setDragging(false);
  // }

  // the train function will iteratively run our optimiser function.
  // train function: running in the Two.js animation loop ~60 times per second
  // optimiser.minimize() automatically adjusts our tf.variable coefficents
  // const train = () => {
  //   tf.tidy(() => {
  //     if (x_vals.length > 0) {
  //       const y = tf.tensor1d(y_vals);

  //       // min the loss by pred from x, y vals
  //       optimiser.minimize(() => loss(predict(x_vals), y));
  //     }
  //   });
  // };

  const setup = (p5, canvasParentRef) => {
    // coefficient variables
    setAfeature(tf.variable(tf.scalar(Math.random(-1, 1))));
    setBfeature(tf.variable(tf.scalar(Math.random(-1, 1))));
    setCfeature(tf.variable(tf.scalar(Math.random(-1, 1))));
    setDfeature(tf.variable(tf.scalar(Math.random(-1, 1))));

    // use parent to render the canvas in this ref
    // (without that p5 will render the canvas outside of your component)
    p5.createCanvas(canvWidth, canvHeight).parent(canvasParentRef);
  };

  const draw = (p5) => {
    p5.background(0);
    p5.stroke(255);
    p5.strokeWeight(8);

    // NOTE: Do not use setState in the draw function or in functions that are executed
    // in the draw function...
    // please use normal variables or class properties for these purposes

    tf.tidy(() => {
      if (x_vals.length > 0) {
        const y = tf.tensor1d(y_vals);

        // minimize the loss by pred from x, y vals
        optimizer.minimize(() => loss(predict(x_vals), y));
      }
    });

    // loop mouse coordinate and plot clicked points
    for (let i = 0; i < x_vals.length; i++) {
      let px = mapRange(x_vals[i], initCoordinate, endCoordinate, 0, canvWidth);
      let py = mapRange(
        y_vals[i],
        initCoordinate,
        endCoordinate,
        canvHeight,
        0
      );
      p5.point(px, py);
    }

    const curveX = [];

    // make an array that has -1, 1 array of values
    for (let x = -1; x < 1.01; x += 0.5) {
      curveX.push(x);
    }

    // clean tensors from predict func
    const ys = tf.tidy(() => predict(curveX));
    // transform tensor into num value
    let curveY = ys.dataSync();
    // clean tensors to avoid meme leaks
    ys.dispose();

    // draw curve
    p5.beginShape();
    p5.noFill();
    p5.stroke(255);
    p5.strokeWeight(2);
    for (let i = 0; i < curveX.length; i++) {
      let x = mapRange(curveX[i], -1, 1, 0, canvWidth);
      let y = mapRange(curveY[i], -1, 1, 0, canvHeight);
      p5.vertex(x, y);
    }
    p5.endShape();

    // draw the line between the points
    p5.strokeWeight(2);
    p5.line(x1, y1, x2, y2);

    console.log(tf.memory().numTensors);
    console.log({ x_vals }, { y_vals });
  };

  return (
    <Sketch
      setup={setup}
      draw={draw}
      mouseClicked={(e) => {
        mousePressed(e.mouseX, e.mouseY);
      }}
      mouseDragged={(e) => {
        onDragged(e);
      }}
    />
  );
}

// https://thecodingtrain.com/CodingChallenges/105-polynomial-regression-tfjs.html
// https://github.com/atorov/react-hooks-p5js/blob/master/src/components/P5Wrapper/index.jsx
// https://www.npmjs.com/package/react-p5

// How can we make this so the degree of the polynomial is interactive as
