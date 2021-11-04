// Equations for quaternion to euler Jacobian
const float JS0 = 2*powf(q2, 2) - 1;
const float JS1 = JS0 + 2*powf(q1, 2);
const float JS2 = q0*q1 + q2*q3;
const float JS3 = 2/(powf(JS1, 2) + 4*powf(JS2, 2));
const float JS4 = JS1*JS3;
const float JS5 = 4*JS2;
const float JS6 = 2/sqrtf(1 - 4*powf(q0*q2 - q1*q3, 2));
const float JS7 = JS0 + 2*powf(q3, 2);
const float JS8 = q0*q3 + q1*q2;
const float JS9 = 2/(powf(JS7, 2) + 4*powf(JS8, 2));
const float JS10 = JS7*JS9;
const float JS11 = 4*JS8;


J(0,0) = -JS4*q1;
J(0,1) = JS3*(-JS1*q0 + JS5*q1);
J(1,1) = -JS6*q3;
J(0,2) = JS3*(-JS1*q3 + JS5*q2);
J(1,2) = JS6*q0;
J(2,2) = JS9*(JS11*q2 - JS7*q1);
J(0,3) = -JS4*q2;
J(1,3) = -JS6*q1;
J(2,3) = JS9*(JS11*q3 - JS7*q0);
