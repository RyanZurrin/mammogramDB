var O = O || {}; // our namespace

O.Loader = function() {

  // constructor
  this.classvariable1 = 'hello';

};


O.Loader.prototype.load = function(param1, param2) {


  return this.classvariable1 + param1 + param2;

};

O.Loader.somestaticmethod = function() {

  // anything without access to 'this'

};
