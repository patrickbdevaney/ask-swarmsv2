import React from 'react';
import { Button } from '@/shared/components/ui/Button';

export default function SwarmsGetStarted() {
  return (
    <section className="py-20 text-center">
      <h3 className="text-3xl font-bold mb-8">Get Started with Swarms</h3>
      <div className="bg-gray-100 p-4 rounded-md inline-block mb-8">
        <code className="text-black">pip install -U swarms</code>
      </div>
      <p className="mb-8">Requires Python 3.10 or above</p>
      <Button size="lg">Read the Docs</Button>
    </section>
  );
}
