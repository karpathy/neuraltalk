var randperm = function(n) {
  var i = n,
      j = 0,
      temp;
  var array = [];
  for(var q=0;q<n;q++)array[q]=q;
  while (i--) {
      j = Math.floor(Math.random() * (i+1));
      temp = array[i];
      array[i] = array[j];
      array[j] = temp;
  }
  return array;
}

var arrContains = function(arr, elt) {
  for(var i=0,n=arr.length;i<n;i++) {
    if(arr[i]===elt) return true;
  }
  return false;
}

var arrUnique = function(arr) {
  var b = [];
  for(var i=0,n=arr.length;i<n;i++) {
    if(!arrContains(b, arr[i])) {
      b.push(arr[i]);
    }
  }
  return b;
}

var renderHSL = function(hsl) { // omg
  var ht = Math.min(360, Math.max(0, hsl[0]));
  var st = Math.min(100, Math.max(0, hsl[1]));
  var lt = Math.min(100, Math.max(0, hsl[2]));
  return 'hsl(' + ht + ',' + st + '%,' + lt + '%)';
}
